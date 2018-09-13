import wget

def download_file(file_link, file_name):
	wget.download(file_link, file_name)
		
download_file('https://maups.github.io/tcv3/data_part1.tar.bz2', 'digits')
