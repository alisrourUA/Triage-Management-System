#this contains that sorts the patients in the queue according to their priority fromm high (1) to low (4)
#it is bubble sorting
def sort_by_priority(patients):
    n = len(patients)
    for i in range(n):
        for j in range(0, n - i - 1):
            if patients[j]["Priority"] > patients[j + 1]["Priority"]:
                patients[j], patients[j + 1] = patients[j + 1], patients[j]
    return patients
