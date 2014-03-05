import cv2
import cv
import numpy as np
import matplotlib
matplotlib.use('gtkcairo')
from matplotlib import pyplot as plt
from scipy import signal as sig

# Verites terrain a afficher
x = [10.414, 12.734, 15.134, 18.414, 24.174, 26.214, 33.454, 37.774, 40.654, 41.894, 48.374, 50.254, 57.414, 60.654, 64.334, 65.774, 68.014, 73.45, 81.934, 85.294, 88.454, 99.934, 106.014, 108.294, 109.414, 118.414, 124.894, 128.654, 132.814, 135.094, 137.094, 137.934, 138.694, 142.654, 147.374, 149.374, 151.334, 173.534, 182.174, 183.814, 185.734, 187.494, 194.494, 195.294, 196.174, 197.454, 200.414, 202.774, 205.094, 207.014, 215.494, 216.614, 218.054, 219.014, 220.254, 220.814, 221.534, 223.814, 224.534, 228.454, 230.854, 231.494, 232.014, 234.014, 235.094, 237.254, 238.894, 243.614, 246.174, 248.854, 250.574, 251.974, 256.134, 259.814, 262.174, 263.614, 269.814, 273.054, 274.934, 281.774, 285.174, 286.534, 293.894, 295.534, 297.134, 298.614, 299.414, 300.974, 305.734, 307.214, 314.174, 315.894, 323.334, 324.454, 329.894, 332.134, 334.134, 336.294, 339.814, 342.334, 343.494, 371.294, 374.774, 375.454, 376.534, 377.814, 383.494, 384.934, 387.094, 390.934, 394.174, 398.414, 401.254, 403.654, 406.494, 408.374, 409.494, 415.094, 417.014, 420.654, 424.694, 426.614, 427.494, 429.374, 430.294, 436.734, 438.614, 442.734, 443.854, 452.014, 454.494, 460.294, 461.454, 466.894, 468.454, 470.374, 471.414, 472.294, 475.054, 492.854, 493.854, 495.814, 498.254, 500.814, 502.454, 505.054, 506.574, 508.134, 509.934, 512.134, 514.054, 516.414, 517.894, 522.454, 523.814, 527.574, 534.174, 537.014, 538.454, 543.414, 544.614, 547.774, 549.614, 552.454, 554.654, 555.774, 557.094, 562.294, 563.334, 566.894, 568.614, 570.654, 574.214, 576.294, 578.174, 579.494, 581.574, 582.934, 586.654, 592.894, 594.694, 596.774, 598.374, 600.054, 602.134, 605.254, 608.294, 611.374, 613.054, 620.734, 623.334, 625.414, 627.734, 630.934, 632.734, 633.494, 634.534, 641.534, 642.374, 645.214, 647.374, 656.454, 658.374, 659.414, 672.494, 674.894, 677.894, 684.734, 690.294, 692.934, 695.254, 698.894, 700.614, 703.614, 705.254, 707.694, 712.574, 721.454, 728.254, 733.934, 737.454, 742.334, 745.374, 747.454, 749.374, 752.174, 754.654, 758.774, 765.214, 772.334, 780.094, 790.494, 791.734, 793.294, 796.934, 798.254, 801.134, 802.934, 804.854, 805.774, 806.534, 807.494, 809.134, 820.494, 821.214, 824.174, 826.574, 827.894, 830.574, 833.214, 835.374, 836.894, 838.174, 839.614, 840.494, 843.054, 844.934, 847.654, 855.014, 858.574, 863.934, 868.494, 871.974, 875.614, 882.734, 884.974, 886.254, 888.134, 890.574, 901.774, 930.654, 933.454, 940.094, 940.894, 942.974, 946.494, 952.134, 954.134, 956.134, 958.494, 961.094, 964.134, 967.014, 967.934, 978.334, 988.014, 991.854, 995.534, 1000.33, 1004.77, 1007.77, 1009.13, 1010.77, 1011.93, 1015.05, 1017.53, 1023.37, 1026.25, 1027.73, 1030.09, 1034.29, 1042.09, 1044.93, 1046.85, 1057.33, 1058.25, 1060.97, 1067.49, 1068.77, 1070.13, 1071.53, 1073.33, 1075.65, 1080.17, 1083.61, 1088.33, 1089.61, 1090.65, 1095.85, 1097.45, 1100.81, 1101.81, 1105.89, 1125.57, 1145.09, 1163.25, 1165.21, 1170.57, 1175.53, 1180.61, 1182.73, 1183.93, 1186.73, 1191.05, 1192.73, 1194.93, 1200.49, 1205.21, 1210.17, 1211.09, 1218.13, 1222.41, 1223.65, 1226.09, 1227.53, 1228.57, 1259.37, 1262.65, 1264.41, 1265.77, 1271.09, 1271.65, 1275.21, 1280.69, 1284.21, 1288.45, 1291.77]

for i,val in enumerate(x) :
	x[i] = round(val*25,0) 

capture = cv2.VideoCapture('TBBT.mp4')

taille =  int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
pasComp = 10
t=[0]*pasComp # Initialise le tableau des dix images que nous allons traiter (Le pas de comparaison : cad on compare a +4/-5 images de l'image ou on est)
for s in range(pasComp):
    val , temp = capture.read()
    if val : t[s] = temp
    else : t[s] = None

dis = [[0 for u in range(taille) ] for v in range(3) ] # 0 = b, 1 = g, 2 = r
color = ('b','g','r')
i = pasComp/2 
while t[pasComp-1] is not None and i < taille: #Tant qu'il n'y a pas de True dans la table	
	for j,col in enumerate(color): 
		histrM5 = cv2.calcHist([t[0]],[j],None,[50],[0,256]) 
		histrP5 = cv2.calcHist([t[pasComp-1]],[j],None,[50],[0,256]) 
		histrP5 = histrP5/np.sum(histrP5)
		histrM5 = histrM5/np.sum(histrM5)
		dist = cv2.compareHist(histrM5, histrP5, cv.CV_COMP_CHISQR) 
		dis[j][i] = dist
	i += 1
	for k in range(pasComp-1):
		t[k] = t[k+1]	
	val , temp = capture.read()
	if val : t[pasComp-1] = temp
	else : t[pasComp-1] = None

# Ajoute les 3 courbes des distances
#plt.plot(dis[0], 'b') 
#plt.plot(dis[1], 'g') 
#plt.plot(dis[2], 'r') 

# Ajoute les barres pour visualiser les verites terrain
#plt.hist(x, taille, facecolor='g')

# Parametres du graphe
#plt.title('TBBT Dist') #titre
#plt.ylim([0,3]) #taille des ordonnees
#plt.xlabel('Frames')
#plt.ylabel('Distances')
#plt.ion() # permet de continuer a utiliser le terminal
#plt.show() # Affiche

# Recupere les maximum locaux des courbes
maxR = sig.argrelmax(np.array(dis[2]),  order = 50)
maxV = sig.argrelmax(np.array(dis[1]), order = 50)
maxB = sig.argrelmax(np.array(dis[0]), order = 50)

# recupere les valeurs des distances correspondantes
valR = np.take(dis[2], maxR)
valV = np.take(dis[1], maxV)
valB = np.take(dis[0], maxB)

# Resultats -> detections en trop 
#   Verites terrain |      |       |  260  |  318  |  378  |  460  |       |  604  |  655  |       |  836  |  944
#   Rouge    2      |  32  |       |  257  |  318  |  381  |  462  |       |  599  |  658  |  710  |  831  |  946
#   Vert     1      |  31  |       |  259  |  314  |  374  |  455  |  547  |  599  |  650  |  712  |  839  |  942
#   Bleu     0      |  31  |  115  |  260  |  313  |  377  |  455  |       | 	   |  655  |  710  |  839  |  947

# tempsR            1735,0          1,0122  7,7945  0,3997  0,7019          0,4669  0,3536  0,0101  0,1492   0,2174
# tempsV            1693,9          0,3631  1.2889  0,3331  0,4500  0,0360  0,3090  0,8549  0,0138  0,0481   0,1725
# tempsB            2269,7  0,1113  0,4400  1.4021  0,3003  0,8435                  1,1269  0,0131  0,0691   0,1118

# Enleve les valeur inferieur a 0.04 -> sinon detecte trop de coupures (Une fonction existe peutetre -> pas trouvee)
tempB = []
tempR = []
tempV = []
incr = 0
while incr < valR.size or incr < valB.size or incr < valV.size :
	if incr < valR.size:
		if valR.item(incr) > 0.02 :
			tempR.append(maxR[0][incr])
	if incr < valV.size:
		if valV.item(incr) > 0.02 :
			tempV.append(maxV[0][incr])
	if incr < valB.size:
		if valB.item(incr) > 0.02 :
			tempB.append(maxB[0][incr])

	incr+=1

# et on remet dans les maxR/V/B
maxR = np.array(tempR)
maxV = np.array(tempV)
maxB = np.array(tempB)

tempPlans = [] # [[numeroFrame,nbVotes, frameMoyenne], ...]
# Systeme de votes : si deux ou trois couleurs detectent une coupure -> il y en a une. Si qu'une, ou l'oublie (Distance max acceptee pour quon parle de la meme coupure : 10)
# A mettre dans lautre boucle while peut-etre plus tard
i = 0
j = 0
# Initialisation avec les votes du rouge
for i, val in enumerate(maxR) :
	tempPlans.append([val,1, val])
# Vote du vert
for i, val1 in enumerate(maxV) :
	for j, val2 in enumerate(tempPlans):
		tmp = val2[0] - val1
		if tmp < 10 and tmp > -10:
			val2[1] +=1
			val2[2] += val1
# Vote du Bleu
for i, val1 in enumerate(maxB) :
	for j, val2 in enumerate(tempPlans):
		tmp = val2[0] - val1
		if tmp < 10 and tmp > -10:
			val2[1] +=1
			val2[2] += val1

plans = []
# Calcule de la frame moyenne
for j, val2 in enumerate(tempPlans):
	plans.append(val2[2]/val2[1])

# Calcul du pourcentage de reussite sur la segmentation
rates = 0
reussi = 0
for i, val1 in enumerate(plans):
	for j, val2 in enumerate(x):
		if val1-val2 > -10 and val1-val2 < 10:
			reussi += 1
			break
print('Reussis ' + str(reussi) + ' taille verites terrain : ' + str(len(x)) + ' taille resultats : '  + str(len(plans)))
print('Pourcentage réussite : ' + str(round(reussi*100./len(x),1)) + '%')

