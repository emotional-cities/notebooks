import os
import sys
import glob

# Get current directory and user name
cdir = os.getcwd()
user = os.environ.get('USER') or os.environ.get('UserName')

# Define paths based on user
if user == 'joaop':  # Personal computer
	try:
		root = r'Z:\Exp_4-outdoor_walk\lisbon'  # LAN
		os.chdir(root)
		os.chdir(cdir)
	except:
		root = r'D:\Joao\Exp_4-outdoor_walk\lisbon'  # Local
	scripts = r'C:\Users\joaop\git\JoaoAmaro2001\WorkRepo'
elif user == 'Administrator':  # MSI computer
	try:
		root = r'Z:\Exp_4-outdoor_walk\lisbon'  # LAN
		os.chdir(root)
		os.chdir(cdir)
	except:
		root = r'I:\Joao\Exp_4-outdoor_walk\lisbon'  # Local
	scripts = r'C:\git\JoaoAmaro2001\WorkRepo'
elif user == 'NGR_FMUL':  # MSI computer
	try:
		root = r'Z:\Exp_4-outdoor_walk\lisbon'  # LAN
		os.chdir(root)
		os.chdir(cdir)
	except:
		root = r'I:\Joao\Exp_4-outdoor_walk\lisbon'  # Local
	scripts = r'C:\github\JoaoAmaro2001\WorkRepo'
else:
	sys.exit('The directories for the input and output data could not be found')

# Define other paths
sourcedata = os.path.join(root, 'sourcedata')
bidsroot = os.path.join(root, 'bids')
results = os.path.join(root, 'results')
derivatives = os.path.join(root, 'derivatives')

# Add scripts to Python path
for path in glob.glob(os.path.join(scripts, '**'), recursive=True):
	if os.path.isdir(path):
		sys.path.append(path)
