from nltk.tokenize import sent_tokenize, word_tokenize
with open("./brown.txt", 'r', encoding="utf8") as f:
    data=f.read()
    data.replace("<", " ")
    data.replace(">", " ")
    data.replace("@", " ")
    data.replace("#", " ")
    data.replace("<", " ")
    
    lines=sent_tokenize(data)
    count=len(lines)
    print(count)
    for index, line in enumerate(lines):
        if index<35000:
            with open("./train.txt", "a") as f1:
                f1.write(line.strip().replace("\n", " "))
                f1.write("\n")
        elif index<45000:
            with open("./valid.txt", "a") as f1:
                f1.write(line.strip().replace("\n", " "))
                f1.write("\n")

        else:
            with open("./test.txt", "a") as f1:
                f1.write(line.strip().replace("\n", " "))
                f1.write("\n")
