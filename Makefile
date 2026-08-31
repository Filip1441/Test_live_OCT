CXX = clang++
CXXFLAGS = -std=c++20 -O3 -g -Wall -Wno-deprecated-declarations -DACCELERATE_NEW_LAPACK -DIMGUI_IMPL_OSX_DISABLE_GAMEPAD
INCLUDES = -Iinclude -Ivendor/imgui -Ivendor/implot -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lhdf5 -lzstd -lz \
          -framework Cocoa \
          -framework Metal \
          -framework MetalKit \
          -framework Accelerate \
          -framework QuartzCore \
          -framework ImageIO \
          -framework CoreGraphics \
          -framework GameController

SRC_CPP = src/hdf5_loader.cpp \
          src/dataset_loader.cpp \
          src/colormaps.cpp \
          src/producer_thread.cpp \
          src/cpu_compute_engine.cpp \
          src/processor_thread.cpp \
          src/miguel_unwrapper.cpp \
          vendor/imgui/imgui.cpp \
          vendor/imgui/imgui_draw.cpp \
          vendor/imgui/imgui_tables.cpp \
          vendor/imgui/imgui_widgets.cpp \
          vendor/implot/implot.cpp \
          vendor/implot/implot_items.cpp

SRC_MM = src/metal_compute_engine.mm \
         src/volume3d_engine.mm \
         src/file_dialog.mm \
         src/app_gui.mm \
         src/main.mm \
         src/python_api.mm \
         vendor/imgui/imgui_impl_osx.mm \
         vendor/imgui/imgui_impl_metal.mm

OBJ_DIR = build
BIN_DIR = bin

OBJS_CPP = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC_CPP))
OBJS_MM  = $(patsubst %.mm, $(OBJ_DIR)/%.o, $(SRC_MM))
OBJS = $(OBJS_CPP) $(OBJS_MM)

TARGET = $(BIN_DIR)/oct_simulator
TEST_TARGET = $(BIN_DIR)/test_oct_core
VALIDATE_TARGET = $(BIN_DIR)/validate_phase_oct
RECON_TARGET = $(BIN_DIR)/phase_oct_recon
LIB_TARGET = $(BIN_DIR)/libphase_oct.dylib

all: $(TARGET) $(TEST_TARGET) $(VALIDATE_TARGET) $(RECON_TARGET) $(LIB_TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $@
	@echo "Build successful: $(TARGET)"

TEST_OBJS = $(OBJ_DIR)/tests/test_oct_core.o \
            $(OBJ_DIR)/src/hdf5_loader.o \
            $(OBJ_DIR)/src/dataset_loader.o \
            $(OBJ_DIR)/src/colormaps.o \
            $(OBJ_DIR)/src/producer_thread.o \
            $(OBJ_DIR)/src/cpu_compute_engine.o \
            $(OBJ_DIR)/src/processor_thread.o \
            $(OBJ_DIR)/src/metal_compute_engine.o

$(TEST_TARGET): $(TEST_OBJS) | $(BIN_DIR)
	@echo "Linking $(TEST_TARGET)..."
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(LDFLAGS) -o $@
	@echo "Build successful: $(TEST_TARGET)"

VALIDATE_OBJS = $(OBJ_DIR)/tools/validate_phase_oct.o \
                $(OBJ_DIR)/src/hdf5_loader.o \
                $(OBJ_DIR)/src/volume3d_engine.o \
                $(OBJ_DIR)/src/miguel_unwrapper.o \
                $(OBJ_DIR)/src/colormaps.o \
                $(OBJ_DIR)/src/file_dialog.o \
                $(OBJ_DIR)/vendor/imgui/imgui.o \
                $(OBJ_DIR)/vendor/imgui/imgui_draw.o \
                $(OBJ_DIR)/vendor/imgui/imgui_tables.o \
                $(OBJ_DIR)/vendor/imgui/imgui_widgets.o \
                $(OBJ_DIR)/vendor/implot/implot.o \
                $(OBJ_DIR)/vendor/implot/implot_items.o \
                $(OBJ_DIR)/vendor/imgui/imgui_impl_osx.o \
                $(OBJ_DIR)/vendor/imgui/imgui_impl_metal.o

$(VALIDATE_TARGET): $(VALIDATE_OBJS) | $(BIN_DIR)
	@echo "Linking $(VALIDATE_TARGET)..."
	$(CXX) $(CXXFLAGS) $(VALIDATE_OBJS) $(LDFLAGS) -o $@
	@echo "Build successful: $(VALIDATE_TARGET)"

RECON_OBJS = $(OBJ_DIR)/src/production_recon_main.o \
             $(OBJ_DIR)/src/hdf5_loader.o \
             $(OBJ_DIR)/src/volume3d_engine.o \
             $(OBJ_DIR)/src/miguel_unwrapper.o \
             $(OBJ_DIR)/src/colormaps.o \
             $(OBJ_DIR)/src/file_dialog.o \
             $(OBJ_DIR)/vendor/imgui/imgui.o \
             $(OBJ_DIR)/vendor/imgui/imgui_draw.o \
             $(OBJ_DIR)/vendor/imgui/imgui_tables.o \
             $(OBJ_DIR)/vendor/imgui/imgui_widgets.o \
             $(OBJ_DIR)/vendor/implot/implot.o \
             $(OBJ_DIR)/vendor/implot/implot_items.o \
             $(OBJ_DIR)/vendor/imgui/imgui_impl_osx.o \
             $(OBJ_DIR)/vendor/imgui/imgui_impl_metal.o

LIB_OBJS = $(OBJ_DIR)/src/python_api.o \
           $(OBJ_DIR)/src/hdf5_loader.o \
           $(OBJ_DIR)/src/volume3d_engine.o \
           $(OBJ_DIR)/src/miguel_unwrapper.o \
           $(OBJ_DIR)/src/colormaps.o \
           $(OBJ_DIR)/src/file_dialog.o \
           $(OBJ_DIR)/vendor/imgui/imgui.o \
           $(OBJ_DIR)/vendor/imgui/imgui_draw.o \
           $(OBJ_DIR)/vendor/imgui/imgui_tables.o \
           $(OBJ_DIR)/vendor/imgui/imgui_widgets.o \
           $(OBJ_DIR)/vendor/implot/implot.o \
           $(OBJ_DIR)/vendor/implot/implot_items.o \
           $(OBJ_DIR)/vendor/imgui/imgui_impl_osx.o \
           $(OBJ_DIR)/vendor/imgui/imgui_impl_metal.o

$(RECON_TARGET): $(RECON_OBJS) | $(BIN_DIR)
	@echo "Linking $(RECON_TARGET)..."
	$(CXX) $(CXXFLAGS) $(RECON_OBJS) $(LDFLAGS) -o $@
	@echo "Build successful: $(RECON_TARGET)"

$(LIB_TARGET): $(LIB_OBJS) | $(BIN_DIR)
	@echo "Linking $(LIB_TARGET)..."
	$(CXX) $(CXXFLAGS) -shared $(LIB_OBJS) $(LDFLAGS) -o $@
	@echo "Build successful: $(LIB_TARGET)"

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: %.mm | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fobjc-arc -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

validate: $(VALIDATE_TARGET)
	./$(VALIDATE_TARGET)

recon: $(RECON_TARGET)
	./$(RECON_TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run test validate recon
