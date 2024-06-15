# build essentials
#! Uncomment if failed 
# sudo apt update
# sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
# libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
# libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip -y

## build
# pip install --user wheel==0.38.0 setuptools==65.5.0 six
# conda install anaconda::py-boost -y

### dependences
# cd /usr/lib/x86_64-linux-gnu/
# [ ! -e "libGL.so" ] && sudo ln -s libGL.so.1 libGL.so
# [ ! -e "libEGL.so" ] && sudo ln -s libEGL.so.1 libEGL.so
# [ ! -e "libOpenGL.so" ] && sudo ln -s libOpenGL.so.0 libOpenGL.so
# cd -

### install
# MARK: gfootball does not support highly parallel reading, install it locally on each machine!
pip uninstall gfootball -y
pip install gfootball

(
    cd zsceval/envs/grf/
    GFootballPath=$(pip show gfootball | grep Location | cut -d' ' -f2)
    ScenariosPath="$GFootballPath/gfootball/scenarios"
    cp scenarios/academy_3_vs_1_with_keeper_superhard.py $ScenariosPath/academy_3_vs_1_with_keeper.py
    cd -
)

### test
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
python3 -c "import gfootball.env as football_env; env = football_env.create_environment('academy_3_vs_1_with_keeper_superhard'); print(env.reset()); print(env.step([0]))"