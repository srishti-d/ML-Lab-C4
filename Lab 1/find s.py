import csv
a=[]
with open ('/kaggle/input/dataset/data.csv','r') as csvfile:
    for row in csv.reader(csvfile)
        a.append(row)
        print(a)
    print("The total number of training instance are:", len(a))
    num_attribute=len(a[0])-1
    print("Initial hypothesis: ")
    hypothesis=['0']*num_attribute
    print(hypothesis)
    for i in range (0, len(a)):
        if a[i][num_attribute]=='yes':
            for j in range(0, num_attribute):
                if hypothesis[j]=a[i][j] or hypothesis[j]='0':
                    hypothesis[j]=a[i][j]
                else:
                    hypothesis[j]='?'
            print("Hypothesis for training instance}} is:" .format(i+1), hypothesis)
    print("Maximally specific hypothesis for the training instances is: ")
    print(hypothesis)
