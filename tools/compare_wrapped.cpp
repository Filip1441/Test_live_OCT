#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include "volume3d_engine.hpp"
#include "hdf5_loader.hpp"
#include "miguel_unwrapper.hpp"

using namespace oct;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <obj.hdf5> <ref.hdf5>\n";
        return 1;
    }
    
    std::string obj_file = argv[1];
    std::string ref_file = argv[2];
    
    DatasetLoader loader;
    auto obj_vol = loader.load(obj_file);
    auto ref_vol = loader.load(ref_file);
    
    Volume3DEngine engine(obj_vol->width, obj_vol->height, obj_vol->num_lambdas);
    engine.set_wavelength_range(803.0f, 878.0f);
    
    // Uruchamiamy pipeline MATLAB Exact
    engine.process_volume(
        obj_vol->data.data(),
        ref_vol->data.data(),
        1.52f, // refractive index
        PipelineMode::Matlab_Exact_FFT, // mode 4
        false // no detrend
    );
    
    // Ale chcemy owiniętą fazę (wrapped)! W silniku zapisywana jest do ph_reconstructed po odpakowaniu.
    // Aby dostać wrapped, możemy odwrócić unwrap (mod 2pi) albo po prostu wyciągnąć fazę jeszcze raz.
    // Zróbmy prosty dump: zapiszmy wynik z C++.
    
    // Zamiast modyfikować kod silnika, po prostu skopiuję wyciągniętą fazę przed unwrapem
    // Ale w engine.mm unwrap robi to in-place na std::vector<float> ph_unwrapped.
    // Zmodyfikujmy na chwilę engine.mm z zewnątrz? Nie, po prostu zrobimy to w pythonie za pomocą modulo 2pi.
    return 0;
}
