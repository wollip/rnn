import csv

with open("c:\\users\\jason\\desktop\\python\\rnn\\culledList.csv", 'r') as file:
	reader = csv.reader(file)
	dataList = list(reader)
dataList = [[int(anumber) for anumber in alist] for alist in dataList if len(alist) is not 0]

#print(dataList[:10])

correct, incorrect = 0, 0
for alist in dataList:
	print("\ndata", alist)
	predict = rnn()
	output = predict.predict(alist)
	print("output", output)
	if alist[-1] == round(output[len(alist)-1]):
		print("correct")
		correct += 1
	else:
		print("incorrect")
		incorrect += 1

print("finished running")
print("correct:", correct, "incorrect:", incorrect)
