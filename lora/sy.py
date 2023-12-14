import torch

latents = torch.randn((2,4,64,64))
good_latents = torch.randn((2,4,64,64))
total_batch = latents.shape[0]

train_indexs = []
test_indexs = []
for i in range(total_batch):
    bad_latent = latents[i,:,:,:]
    good_latent = good_latents[i,:,:,:]
    if torch.equal(bad_latent, good_latent):
        train_indexs.append(i)
    else :
        test_indexs.append(i)
print(f"train_indexs: {train_indexs}")
print(f"test_indexs: {test_indexs}")

train_latents = latents[train_indexs,:,:,:]

test_latents = latents[test_indexs,:,:,:]
test_good_latents = good_latents[test_indexs,:,:,:]
print(f'train_latents: {train_latents.shape}')
print(f'test_latents: {test_latents.shape}')
print(f'test_good_latents: {test_good_latents.shape}')
