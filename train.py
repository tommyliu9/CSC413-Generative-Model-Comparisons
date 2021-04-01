import click
import os
import sys
import requests
import zipfile
import torch

class ModelArgs(dict):
    def __init__(self, *args, **kwargs):
      super(ModelArgs, self).__init__(*args, **kwargs)
      self.__dict__ = self

def download_data_from_url():
	"""Download data set."""
	path = os.getcwd() + '/data/'

	#Make directory for dataset
	try:
		os.mkdir(path) 
	except OSError:
		sys.exit(f'Creation of directory {path} failed')

 	# Download dataset
	url = 'https://zenodo.org/record/1117372/files/musdb18.zip?download=1'
	filename = path + 'musdb18.zip'
	r = requests.get(url, stream=True)
	with open(filename, 'wb') as f:
		for chunk in r.iter_content(chunk_size=128):
			f.write(chunk)

	with zipfile.ZipFile(filename, 'r') as zipf:
		zipf.extract(path)

@click.command()
@click.option('--download-data/--no-download-data',
 	default=False, help='Download data set, defaults to false.')
@click.option('--epochs', default=100, 
	help='Number of training epochs.')
@click.option('--lr', default=1e-3, help='Initial learning rate.')
@click.option('--cuda/--no-cuda', default=True,
	help='Train on GPU if available.')
@click.option('--beta', nargs=2, type=(float, float), 
	help='Beta range for Adam optimizer.')
def main(download_data, epochs, lr, cuda, beta):
	if download_data:
		download_data_from_url()

	if cuda:
		if not torch.cuda.is_available():
			print("Cuda not available")
			return
		
		device = torch.device('cuda')

	args = ModelArgs({
		'epochs': epochs,
		'lr': lr,
		'beta': beta,
		'device': device
	})
	print(epochs, lr, beta, device, download_data)
	return args


if __name__ == '__main__':
	main()
