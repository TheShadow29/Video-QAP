CUR_DIR=$(pwd)
DATA_ROOT=${2:-$CUR_DIR}

mkdir -p $DATA_ROOT/anet

function gdrive_download () {
	CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
	rm -rf /tmp/cookies.txt
}

function dwn_asrl_ch_qa_files() {
    gdrive_download "1PqHYty4D71dakC1a95p4PbAr6WXYYCYq" asrl_ch_qa_files.zip
}

function dwn_anet_feats() {
    # RGB motion feats
    # Courtesy of Louwei Zhou, obtained from the repository:
    # https://github.com/facebookresearch/grounded-video-description/blob/master/tools/download_all.sh
    cd $DATA_ROOT/anet
    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz
    tar -xvzf rgb_motion_1d.tar.gz && rm rgb_motion_1d.tar.gz

    gdrive_download "14GTjt3wuifK6GhaTsRYxMVXhBc4rREk2" anet_barebones.zip
}

function dwn_ch_feats() {
    cd $DATA_ROOT/charades
    gdrive_download "1bJgzyVue8GhbaImjiVhfNgHk3O9d8Apc" charades_stuff.zip
}