hist= {} 
for x in selected["eth.src"].values: 
    hist[x]= hist.get(x,0) +1
print(hist)