cmake -B out/Release -S . -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake -DCMAKE_BUILD_TYPE=Release
cd out/Release
cmake --build . --config Release
cd ../..