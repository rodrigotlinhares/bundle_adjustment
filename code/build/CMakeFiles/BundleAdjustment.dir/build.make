# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rodrigolinhares/code/bundle_adjustment/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rodrigolinhares/code/bundle_adjustment/code/build

# Include any dependencies generated for this target.
include CMakeFiles/BundleAdjustment.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BundleAdjustment.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BundleAdjustment.dir/flags.make

CMakeFiles/BundleAdjustment.dir/main.cpp.o: CMakeFiles/BundleAdjustment.dir/flags.make
CMakeFiles/BundleAdjustment.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BundleAdjustment.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BundleAdjustment.dir/main.cpp.o -c /home/rodrigolinhares/code/bundle_adjustment/code/main.cpp

CMakeFiles/BundleAdjustment.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BundleAdjustment.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrigolinhares/code/bundle_adjustment/code/main.cpp > CMakeFiles/BundleAdjustment.dir/main.cpp.i

CMakeFiles/BundleAdjustment.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BundleAdjustment.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrigolinhares/code/bundle_adjustment/code/main.cpp -o CMakeFiles/BundleAdjustment.dir/main.cpp.s

CMakeFiles/BundleAdjustment.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/BundleAdjustment.dir/main.cpp.o.requires

CMakeFiles/BundleAdjustment.dir/main.cpp.o.provides: CMakeFiles/BundleAdjustment.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/BundleAdjustment.dir/build.make CMakeFiles/BundleAdjustment.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/BundleAdjustment.dir/main.cpp.o.provides

CMakeFiles/BundleAdjustment.dir/main.cpp.o.provides.build: CMakeFiles/BundleAdjustment.dir/main.cpp.o

CMakeFiles/BundleAdjustment.dir/ba.cpp.o: CMakeFiles/BundleAdjustment.dir/flags.make
CMakeFiles/BundleAdjustment.dir/ba.cpp.o: ../ba.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BundleAdjustment.dir/ba.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BundleAdjustment.dir/ba.cpp.o -c /home/rodrigolinhares/code/bundle_adjustment/code/ba.cpp

CMakeFiles/BundleAdjustment.dir/ba.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BundleAdjustment.dir/ba.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrigolinhares/code/bundle_adjustment/code/ba.cpp > CMakeFiles/BundleAdjustment.dir/ba.cpp.i

CMakeFiles/BundleAdjustment.dir/ba.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BundleAdjustment.dir/ba.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrigolinhares/code/bundle_adjustment/code/ba.cpp -o CMakeFiles/BundleAdjustment.dir/ba.cpp.s

CMakeFiles/BundleAdjustment.dir/ba.cpp.o.requires:
.PHONY : CMakeFiles/BundleAdjustment.dir/ba.cpp.o.requires

CMakeFiles/BundleAdjustment.dir/ba.cpp.o.provides: CMakeFiles/BundleAdjustment.dir/ba.cpp.o.requires
	$(MAKE) -f CMakeFiles/BundleAdjustment.dir/build.make CMakeFiles/BundleAdjustment.dir/ba.cpp.o.provides.build
.PHONY : CMakeFiles/BundleAdjustment.dir/ba.cpp.o.provides

CMakeFiles/BundleAdjustment.dir/ba.cpp.o.provides.build: CMakeFiles/BundleAdjustment.dir/ba.cpp.o

CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o: CMakeFiles/BundleAdjustment.dir/flags.make
CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o: ../ba_illum.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o -c /home/rodrigolinhares/code/bundle_adjustment/code/ba_illum.cpp

CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrigolinhares/code/bundle_adjustment/code/ba_illum.cpp > CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.i

CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrigolinhares/code/bundle_adjustment/code/ba_illum.cpp -o CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.s

CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.requires:
.PHONY : CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.requires

CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.provides: CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.requires
	$(MAKE) -f CMakeFiles/BundleAdjustment.dir/build.make CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.provides.build
.PHONY : CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.provides

CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.provides.build: CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o

CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o: CMakeFiles/BundleAdjustment.dir/flags.make
CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o: ../MOSAIC.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o -c /home/rodrigolinhares/code/bundle_adjustment/code/MOSAIC.cpp

CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrigolinhares/code/bundle_adjustment/code/MOSAIC.cpp > CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.i

CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrigolinhares/code/bundle_adjustment/code/MOSAIC.cpp -o CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.s

CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.requires:
.PHONY : CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.requires

CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.provides: CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.requires
	$(MAKE) -f CMakeFiles/BundleAdjustment.dir/build.make CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.provides.build
.PHONY : CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.provides

CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.provides.build: CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o

CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o: CMakeFiles/BundleAdjustment.dir/flags.make
CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o: ../tracking_aux.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o -c /home/rodrigolinhares/code/bundle_adjustment/code/tracking_aux.cpp

CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrigolinhares/code/bundle_adjustment/code/tracking_aux.cpp > CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.i

CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrigolinhares/code/bundle_adjustment/code/tracking_aux.cpp -o CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.s

CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.requires:
.PHONY : CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.requires

CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.provides: CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.requires
	$(MAKE) -f CMakeFiles/BundleAdjustment.dir/build.make CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.provides.build
.PHONY : CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.provides

CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.provides.build: CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o

CMakeFiles/BundleAdjustment.dir/utils.cpp.o: CMakeFiles/BundleAdjustment.dir/flags.make
CMakeFiles/BundleAdjustment.dir/utils.cpp.o: ../utils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BundleAdjustment.dir/utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BundleAdjustment.dir/utils.cpp.o -c /home/rodrigolinhares/code/bundle_adjustment/code/utils.cpp

CMakeFiles/BundleAdjustment.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BundleAdjustment.dir/utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrigolinhares/code/bundle_adjustment/code/utils.cpp > CMakeFiles/BundleAdjustment.dir/utils.cpp.i

CMakeFiles/BundleAdjustment.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BundleAdjustment.dir/utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrigolinhares/code/bundle_adjustment/code/utils.cpp -o CMakeFiles/BundleAdjustment.dir/utils.cpp.s

CMakeFiles/BundleAdjustment.dir/utils.cpp.o.requires:
.PHONY : CMakeFiles/BundleAdjustment.dir/utils.cpp.o.requires

CMakeFiles/BundleAdjustment.dir/utils.cpp.o.provides: CMakeFiles/BundleAdjustment.dir/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/BundleAdjustment.dir/build.make CMakeFiles/BundleAdjustment.dir/utils.cpp.o.provides.build
.PHONY : CMakeFiles/BundleAdjustment.dir/utils.cpp.o.provides

CMakeFiles/BundleAdjustment.dir/utils.cpp.o.provides.build: CMakeFiles/BundleAdjustment.dir/utils.cpp.o

# Object files for target BundleAdjustment
BundleAdjustment_OBJECTS = \
"CMakeFiles/BundleAdjustment.dir/main.cpp.o" \
"CMakeFiles/BundleAdjustment.dir/ba.cpp.o" \
"CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o" \
"CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o" \
"CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o" \
"CMakeFiles/BundleAdjustment.dir/utils.cpp.o"

# External object files for target BundleAdjustment
BundleAdjustment_EXTERNAL_OBJECTS =

BundleAdjustment: CMakeFiles/BundleAdjustment.dir/main.cpp.o
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/ba.cpp.o
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/utils.cpp.o
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/build.make
BundleAdjustment: /usr/local/lib/libopencv_videostab.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_video.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_ts.a
BundleAdjustment: /usr/local/lib/libopencv_superres.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_stitching.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_photo.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_ocl.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_objdetect.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_nonfree.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_ml.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_legacy.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_imgproc.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_highgui.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_gpu.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_flann.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_features2d.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_core.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_contrib.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_calib3d.so.2.4.10
BundleAdjustment: /usr/lib/x86_64-linux-gnu/libGLU.so
BundleAdjustment: /usr/lib/x86_64-linux-gnu/libGL.so
BundleAdjustment: /usr/lib/x86_64-linux-gnu/libSM.so
BundleAdjustment: /usr/lib/x86_64-linux-gnu/libICE.so
BundleAdjustment: /usr/lib/x86_64-linux-gnu/libX11.so
BundleAdjustment: /usr/lib/x86_64-linux-gnu/libXext.so
BundleAdjustment: /usr/local/lib/libopencv_nonfree.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_ocl.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_gpu.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_photo.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_objdetect.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_legacy.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_video.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_ml.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_calib3d.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_features2d.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_highgui.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_imgproc.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_flann.so.2.4.10
BundleAdjustment: /usr/local/lib/libopencv_core.so.2.4.10
BundleAdjustment: CMakeFiles/BundleAdjustment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable BundleAdjustment"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BundleAdjustment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BundleAdjustment.dir/build: BundleAdjustment
.PHONY : CMakeFiles/BundleAdjustment.dir/build

CMakeFiles/BundleAdjustment.dir/requires: CMakeFiles/BundleAdjustment.dir/main.cpp.o.requires
CMakeFiles/BundleAdjustment.dir/requires: CMakeFiles/BundleAdjustment.dir/ba.cpp.o.requires
CMakeFiles/BundleAdjustment.dir/requires: CMakeFiles/BundleAdjustment.dir/ba_illum.cpp.o.requires
CMakeFiles/BundleAdjustment.dir/requires: CMakeFiles/BundleAdjustment.dir/MOSAIC.cpp.o.requires
CMakeFiles/BundleAdjustment.dir/requires: CMakeFiles/BundleAdjustment.dir/tracking_aux.cpp.o.requires
CMakeFiles/BundleAdjustment.dir/requires: CMakeFiles/BundleAdjustment.dir/utils.cpp.o.requires
.PHONY : CMakeFiles/BundleAdjustment.dir/requires

CMakeFiles/BundleAdjustment.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BundleAdjustment.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BundleAdjustment.dir/clean

CMakeFiles/BundleAdjustment.dir/depend:
	cd /home/rodrigolinhares/code/bundle_adjustment/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodrigolinhares/code/bundle_adjustment/code /home/rodrigolinhares/code/bundle_adjustment/code /home/rodrigolinhares/code/bundle_adjustment/code/build /home/rodrigolinhares/code/bundle_adjustment/code/build /home/rodrigolinhares/code/bundle_adjustment/code/build/CMakeFiles/BundleAdjustment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BundleAdjustment.dir/depend
