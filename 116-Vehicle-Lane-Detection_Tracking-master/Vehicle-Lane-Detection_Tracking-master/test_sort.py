list1 = [[2,3,4,5],[20,30,40,50],[200,300,400,500]]
list2 = ['one','two','three']
list3 =[(t[2]-t[0])*(t[3]-t[1]) for t in list1]
print(list3)

sorted_list3, list2 = [list(x) for x in zip(*sorted(zip(list3, list2), key=lambda pair: pair[0], reverse=True))]
sorted_list3, list1 = [list(x) for x in zip(*sorted(zip(list3, list1), key=lambda pair: pair[0], reverse=True))]
print(sorted_list3)
print(list2)
print(list1)

# list3 = zip(list1,list2)
# list3
# #print(list3)
# sorted(list3, key=lambda x: x[1])
# #list3 = sorted(list3, key=lambda t:(t[2]-t[0])*(t[3]-t[1]), reverse=True)
# ilist3 = [i for (i,s) in list3]
# ilist3
# slist3 = [s for (i,s) in list3]
# slist3