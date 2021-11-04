import argparse
import os


def is_tecplot_file(filename):
    return any(filename.endswith(extension) for extension in ['.dat', '.DAT'])


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, help="Path of input files dir")
parser.add_argument('-o', type=str, required=True, help="Path of output files dir")

args = parser.parse_args()
input_path = args.i
output_path = args.o

assert os.path.isdir(input_path), '{:s} is not a valid directory'.format(path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

files = []
for dirpath, _, filename in sorted(os.walk(input_path)):
    for fname in sorted(filename):
        if is_tecplot_file(fname):
            files.append(os.path.join(dirpath, fname))

assert files, '{:s} has no valid tecplot file'.format(input_path)

for i, file in enumerate(files):
    print("Processing {} / {}".format(str(i+1), str(len(files))))
    _, filename = os.path.split(file)
    data = ""
    with open(file, "r") as f:
        for j, line in enumerate(f):
            print("In file {}, line {}".format(filename, str(j+1)))
            line = line.lstrip()
            data += line
    if output_path.endswith('/'):
        outfile = output_path + filename
    else:
        outfile = output_path + "/" + filename

    with open(outfile, "w") as f:
        f.write(data)

print("Completed")
