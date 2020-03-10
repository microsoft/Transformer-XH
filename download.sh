# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

mkdir data
cd data
## hack the Onedrive link for downloading
wget --no-check-certificate "https://onedrive.live.com/download?cid=947DE3D32A1EDA0B&resid=947DE3D32A1EDA0B%21152&authkey=AFWhHL-Dz3-hKXw"
cp download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!152\&authkey\=AFWhHL-Dz3-hKXw hotpot_graph.zip
rm download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!152\&authkey\=AFWhHL-Dz3-hKXw
unzip hotpot_graph.zip
rm hotpot_graph.zip

wget --no-check-certificate "https://onedrive.live.com/download?cid=947DE3D32A1EDA0B&resid=947DE3D32A1EDA0B%21150&authkey=AJk9qmL0sm5v4Fs" 
cp download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!150\&authkey\=AJk9qmL0sm5v4Fs fever_graph.zip
rm download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!150\&authkey\=AJk9qmL0sm5v4Fs
unzip fever_graph.zip
rm fever_graph.zip

cd ..
mkdir experiments
cd experiments

wget --no-check-certificate "https://onedrive.live.com/download?cid=947DE3D32A1EDA0B&resid=947DE3D32A1EDA0B%21149&authkey=AB5K4DEN5-biTbk"
cp download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!149\&authkey\=AB5K4DEN5-biTbk transformer_xh_hotpot.zip
rm cp download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!149\&authkey\=AB5K4DEN5-biTbk
unzip transformer_xh_hotpot.zip
rm transformer_xh_hotpot.zip

wget --no-check-certificate "https://onedrive.live.com/download?cid=947DE3D32A1EDA0B&resid=947DE3D32A1EDA0B%21151&authkey=AET3gvtKjjrwnbU" 
cp download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!151\&authkey\=AET3gvtKjjrwnbU transformer_xh_fever.zip
rm download\?cid\=947DE3D32A1EDA0B\&resid\=947DE3D32A1EDA0B\!151\&authkey\=AET3gvtKjjrwnbU
unzip transformer_xh_fever.zip
rm transformer_xh_fever.zip

cd ..