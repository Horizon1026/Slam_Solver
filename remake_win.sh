if [ ! -d "build" ]; then
    mkdir build
fi

cd build/
rm * -rf
cmake -G "MinGW Makefiles" ..
mingw32-make.exe -j
cd ..