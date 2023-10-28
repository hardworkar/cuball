cmake -B out/Release -S . -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 cd ../.. && exit /b %errorlevel%
cd out/Release
cmake --build . --config Release
if %errorlevel% neq 0 cd ../.. && exit /b %errorlevel%
cd ../..
.\out\Release\Release\cuball.exe .\resources\FREDDYFAZBEAR.stl .\resources\BALL.stl