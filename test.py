def hasSingleCycle(array):
    index = []
    visits = [0 for x in array]
    lastindex = len(array) - 1
    currentvisit = 0
    currentvisitedIndex = 0
    currentVisitedIndex = 0
    #visits[tovisit] += 1
    answer = False
    visited = 0
    #for i in range(0, len(array)):
    while(True):
        currentNumbr = array[currentvisitedIndex]

        if(index.count(currentvisitedIndex)==0):
            visited = visited + 1
            index.append(currentvisitedIndex)

        # d =  [2, -4, 1, -3, -4, 2]
        if (currentvisitedIndex + currentNumbr >= 0):
            if(currentvisitedIndex + currentNumbr >= len(array)):
                currentvisitedIndex = (currentvisitedIndex+currentNumbr)- len(array)
            else:
                currentvisitedIndex = currentvisitedIndex + currentNumbr
        else:
            if (currentvisitedIndex + currentNumbr >= 0):
                currentvisitedIndex = (currentvisitedIndex + currentNumbr)
            else:
                currentvisitedIndex =(len(array))+ (currentvisitedIndex + currentNumbr)





        if(visited ==len(array) and currentvisitedIndex==0):
            answer = True
            break
        if (visited < len(array) and currentvisitedIndex == 0):
            answer = False
            break



    return     answer





k =[10, 11, -6, -23, -2, 3, 88, 909, -26]
d = [2, 3, 1, -4, -4, 2]
a = hasSingleCycle(k)

print(a)