from datasets import PartDataset
import torch

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = 2500)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=True, num_workers=0)

for i,data in enumerate(dataloader,0):
	points,target = data
	print points,points.shape,target
	if i == 4:
		break                                        