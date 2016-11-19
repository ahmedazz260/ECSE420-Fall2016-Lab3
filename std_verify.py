import sys
from output import output

if __name__ == "__main__":
	file_name = sys.argv[1] if len(sys.argv) > 1 else 'result.txt'
	iteration_count = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

	the_sum = 0
	with open(file_name, 'r') as f:
		count = 0
		for line in f:
			line = line.strip()
			if line.endswith(','):
				line = line[:-1]

			the_sum += (float(line) - output[count]) ** 2
			count += 1

	the_sum = the_sum / iteration_count
	print "Std is %s" % the_sum