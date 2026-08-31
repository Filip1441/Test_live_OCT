#include <metal_stdlib>
using namespace metal;

// 3D Full Volume Real-FFT on Apple Silicon GPU
kernel void oct_full_volume3d_fft_kernel(
    device const float* sweep_volume [[buffer(0)]], // [N_lambda, H, W]
    device const float* window       [[buffer(1)]], // [N_lambda]
    texture3d<float, access::write>  volume_tex3d   [[texture(2)]], // (Width, Height, Depth)
    constant uint&      n_lambda     [[buffer(3)]],
    constant uint&      depth_bins   [[buffer(4)]],
    constant uint&      width        [[buffer(5)]],
    constant uint&      height       [[buffer(6)]],
    constant uint&      step_xy      [[buffer(7)]],
    constant uint&      raw_w        [[buffer(8)]],
    constant uint&      raw_h        [[buffer(9)]],
    uint2               pos          [[thread_position_in_grid]]
) {
    uint tx = pos.x;
    uint ty = pos.y;

    if (tx >= width || ty >= height) return;

    uint raw_x = tx * step_xy;
    uint raw_y = ty * step_xy;
    if (raw_x >= raw_w || raw_y >= raw_h) return;

    size_t frame_stride = (size_t)raw_w * raw_h;
    size_t pixel_offset = (size_t)raw_y * raw_w + raw_x;

    // DC Offset removal
    float sum = 0.0f;
    for (uint l = 0; l < n_lambda; ++l) {
        sum += sweep_volume[l * frame_stride + pixel_offset];
    }
    float mean = sum / float(n_lambda);

    // Real-FFT across all depth bins
    for (uint k = 0; k < depth_bins; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        float k_factor = -6.28318530717958647692f * float(k) / float(n_lambda);

        for (uint l = 0; l < n_lambda; ++l) {
            float val = (sweep_volume[l * frame_stride + pixel_offset] - mean) * window[l];
            float angle = k_factor * float(l);
            re += val * cos(angle);
            im += val * sin(angle);
        }

        float mag = sqrt(re * re + im * im) + 1e-6f;
        float db = 8.685889638f * log(mag);
        float norm_val = clamp((db - 20.0f) / 75.0f, 0.0f, 1.0f);

        volume_tex3d.write(float4(norm_val, 0.0, 0.0, 1.0), uint3(tx, ty, k));
    }
}

struct Ray {
    float3 origin;
    float3 dir;
};

bool intersect_box(Ray r, float3 box_min, float3 box_max, thread float& t_near, thread float& t_far) {
    float3 inv_dir = 1.0f / (r.dir + float3(1e-8f));
    float3 t0 = (box_min - r.origin) * inv_dir;
    float3 t1 = (box_max - r.origin) * inv_dir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);

    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far = min(min(tmax.x, tmax.y), tmax.z);

    return (t_far > t_near && t_far > 0.0f);
}

// 3D Volumetric Raymarching with ISO-Surface Gradient Normal Shading & Z-Scale
kernel void oct_raymarch_kernel(
    texture3d<float, access::sample> volume_tex     [[texture(0)]],
    texture2d<float, access::write>  render_target  [[texture(1)]],
    device const uint*               colormap_lut   [[buffer(2)]],
    constant float&                  yaw_rad        [[buffer(3)]],
    constant float&                  pitch_rad      [[buffer(4)]],
    constant float&                  zoom_dist      [[buffer(5)]],
    constant float&                  threshold_norm [[buffer(6)]],
    constant float&                  density_scale  [[buffer(7)]],
    constant float&                  z_stretch      [[buffer(8)]], // Z-axis exaggeration (e.g. 1.0 - 5.0)
    constant uint&                   render_mode    [[buffer(9)]], // 0 = Volume Density, 1 = ISO-Surface Shaded, 2 = 3D Topography
    constant uint&                   num_steps      [[buffer(10)]],
    uint2                            pos            [[thread_position_in_grid]]
) {
    uint width = render_target.get_width();
    uint height = render_target.get_height();

    if (pos.x >= width || pos.y >= height) return;

    float u = (float(pos.x) + 0.5f) / float(width) - 0.5f;
    float v = (float(pos.y) + 0.5f) / float(height) - 0.5f;

    // Camera Orbit Matrix
    float cy = cos(yaw_rad);
    float sy = sin(yaw_rad);
    float cp = cos(pitch_rad);
    float sp = sin(pitch_rad);

    float3 cam_pos = float3(
        zoom_dist * sp * sy,
        zoom_dist * cp,
        zoom_dist * sp * cy
    );

    float3 forward = normalize(float3(0.0f) - cam_pos);
    float3 world_up = float3(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, world_up));
    float3 up = cross(right, forward);

    float3 ray_dir = normalize(forward * 1.5f + right * u + up * (-v));

    Ray ray = { cam_pos, ray_dir };
    float3 box_min = float3(-0.5f, -0.5f, -0.5f);
    float3 box_max = float3( 0.5f,  0.5f,  0.5f);

    float t_near = 0.0f, t_far = 0.0f;
    if (!intersect_box(ray, box_min, box_max, t_near, t_far)) {
        float bg = 0.07f + 0.03f * (1.0f - v);
        render_target.write(float4(bg, bg * 1.15f, bg * 1.4f, 1.0f), pos);
        return;
    }

    t_near = max(0.0f, t_near);
    float step_size = (t_far - t_near) / float(num_steps);
    float t = t_near;

    constexpr sampler tex_sampler(coord::normalized, filter::linear, address::clamp_to_edge);

    float4 accum_color = float4(0.0f);
    float3 light_dir = normalize(float3(0.5f, 0.8f, 0.6f));

    for (uint i = 0; i < num_steps && accum_color.a < 0.98f; ++i) {
        float3 p = ray.origin + ray_dir * (t + 0.5f * step_size);

        // Apply Z-stretch and map [-0.5, 0.5] to [0, 1] texture coordinates
        float3 tex_coord = float3(p.x + 0.5f, p.z + 0.5f, (p.y / max(0.1f, z_stretch)) + 0.5f);

        if (tex_coord.x >= 0.0f && tex_coord.x <= 1.0f &&
            tex_coord.y >= 0.0f && tex_coord.y <= 1.0f &&
            tex_coord.z >= 0.0f && tex_coord.z <= 1.0f) {

            float val = volume_tex.sample(tex_sampler, tex_coord).r;

            if (val > threshold_norm) {
                // Compute Gradient Normal for 3D Surface Shading
                float eps = 0.006f;
                float dx = volume_tex.sample(tex_sampler, tex_coord + float3(eps, 0, 0)).r -
                           volume_tex.sample(tex_sampler, tex_coord - float3(eps, 0, 0)).r;
                float dy = volume_tex.sample(tex_sampler, tex_coord + float3(0, eps, 0)).r -
                           volume_tex.sample(tex_sampler, tex_coord - float3(0, eps, 0)).r;
                float dz = volume_tex.sample(tex_sampler, tex_coord + float3(0, 0, eps)).r -
                           volume_tex.sample(tex_sampler, tex_coord - float3(0, 0, eps)).r;
                float3 normal = normalize(float3(-dx, -dz * z_stretch, -dy) + float3(1e-5f));

                // Diffuse + Specular Lighting
                float diff = max(0.2f, dot(normal, light_dir));
                float3 half_v = normalize(light_dir - ray_dir);
                float spec = pow(max(0.0f, dot(normal, half_v)), 16.0f) * 0.4f;

                // Lookup Colormap based on Depth (Z-height)
                uint lut_idx = clamp(uint(tex_coord.z * 255.0f), 0u, 255u);
                uint rgba = colormap_lut[lut_idx];
                float r = float(rgba & 0xFF) / 255.0f;
                float g = float((rgba >> 8) & 0xFF) / 255.0f;
                float b = float((rgba >> 16) & 0xFF) / 255.0f;
                float3 base_rgb = float3(r, g, b) * diff + float3(spec);

                if (render_mode == 1) {
                    // Mode 1: Sharp ISO-Surface
                    accum_color.rgb = base_rgb;
                    accum_color.a = 1.0f;
                    break;
                } else {
                    // Mode 0: Volumetric Density Alpha Blending
                    float alpha = (val - threshold_norm) / max(0.01f, (1.0f - threshold_norm));
                    alpha = clamp(alpha * density_scale * (step_size * 25.0f), 0.0f, 1.0f);

                    accum_color.rgb += (1.0f - accum_color.a) * base_rgb * alpha;
                    accum_color.a += (1.0f - accum_color.a) * alpha;
                }
            }
        }
        t += step_size;
    }

    float bg = 0.07f + 0.03f * (1.0f - v);
    float3 final_rgb = accum_color.rgb + (1.0f - accum_color.a) * float3(bg, bg * 1.15f, bg * 1.4f);

    render_target.write(float4(final_rgb, 1.0f), pos);
}
