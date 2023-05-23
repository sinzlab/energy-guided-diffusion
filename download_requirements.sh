git clone https://github.com/sinzlab/mei ./lib/mei
git clone https://github.com/dicarlolab/CORnet.git ./lib/CORnet
git clone https://github.com/sacadena/ptrnets.git ./lib/ptrnets
git clone -b model_builder https://github.com/sinzlab/nnvision.git ./lib/nnvision

mkdir "data"
curl https://api.wandb.ai/artifactsV2/gcp-us/sinzlab/QXJ0aWZhY3Q6NDYzNDUyNzgw/0241dae91ce4fc127ded0a1598a4f5bd/target_l2.npy > ./data/target_l2.npy
curl https://api.wandb.ai/artifactsV2/gcp-us/sinzlab/QXJ0aWZhY3Q6NDYzNDUyODA0/b6ba56e1f05dd3ef5dfa27d52262d1ef/75_monkey_test_imgs.npy > ./data/75_monkey_test_imgs.npy
curl https://api.wandb.ai/artifactsV2/gcp-us/sinzlab/QXJ0aWZhY3Q6NDYzNDUyODI3/32ef7776f79ea9895c9df78ed294fe85/pretrained_resnet_unit_correlations.npy > ./data/pretrained_resnet_unit_correlations.npy
curl https://api.wandb.ai/artifactsV2/gcp-us/sinzlab/QXJ0aWZhY3Q6NDYzNDUyODUz/5f7c67355d51a99d2f822f2015891420/data_driven_corr.npy > ./data/data_driven_corr.npy

mkdir "models"
curl https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt > ./models/256x256_diffusion_uncond.pt