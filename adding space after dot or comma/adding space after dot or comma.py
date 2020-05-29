import re
ths = open("fixed.tex", "w")
with open("simple.tex", "r") as file:
    while True:
        try:
            line=file.readline()
            if line=="":break
            if (re.findall("\.[A-Z]",line))!=[]:   
                print(error_list)
                print(line)
                error_list=re.findall("\.[A-Z]",line)
                
                for i in error_list:
                    temp=i
                    temp=temp.replace(".", ". ")
                    line=line.replace(i,temp)
            ths.write(line)
        except:
            continue
ths.close()
