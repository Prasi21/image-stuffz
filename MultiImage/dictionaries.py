dictList = []
dictList.append({"t1": 1, "t2": 2})
dictList.append({"t1": 3, "t2": 4})
dictList.append({"t1": 5, "t2": 6})

if not any(dict['t1'] == 1 for dict in dictList):
    print("1 is a value for t1 in the dict")
else:
    print("Nope")