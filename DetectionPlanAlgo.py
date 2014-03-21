import cv2
import cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal as sig
import pickle as pk

class SSAlgorithm():

	#### Constructeur
	def __init__(self, nomVid) : # int, float
		self.order = 10
		self.seuil = 0.02
		f = open(nomVid + "_segm.pkl","rb")
		self.distMoy = pk.load(f)
		f.close()
		self.nomVideo = nomVid
		self.capture = cv2.VideoCapture(self.nomVideo + ".mp4")
		self.fps = int(self.capture.get(cv.CV_CAP_PROP_FPS))


	### Calcule la distance moyenne
	def calcul_dist(self):
		taille = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
		pasComp = 10
		t=[0]*pasComp # Initialise le tableau des dix images que nous allons traiter (Le pas de comparaison : cad on compare a +4/-5 images de l'image ou on est)
		for s in range(pasComp):
		    val , temp = self.capture.read()
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
			val , temp = self.capture.read()
			if val : t[pasComp-1] = temp
			else : t[pasComp-1] = None

		distMoy = [0 for u in range(taille)]
		for i, val in enumerate(dis[2]) :
			distMoy[i] = val
		for i, val in enumerate(dis[1]) :
			distMoy[i] += val
		for i, val in enumerate(dis[0]) :
			distMoy[i] += val
			distMoy[i] = distMoy[i]/3
		return distMoy

	
	#### Detecte les frontieres
	def detect(self) : # str
		
		
		#distMoy = self.calcul_dist()
		
		# Recupere les maxima locaux de la courbe -> return les index
		maxi = sig.argrelmax(np.array(self.distMoy), order = self.order)
		# recupere les valeurs correspondants des maxi
		valDist = np.take(self.distMoy, maxi)
		# Enleve les valeur inferieur a 0.02 -> sinon detecte trop de coupures (Une fonction existe peutetre -> pas trouvee)
		temp = []
		incr = 0
		while incr < valDist.size :
			if valDist.item(incr) > self.seuil :
				temp.append(maxi[0][incr])
			incr+=1
		# et on remet dans les maxR/V/B
		maxi = np.array(temp)
            	return maxi

	
	#### Calcul le nombre de frontieres bien detectees
	def reussis(self, maxi, veritesTerrain) : # [], []
		# Calcul du pourcentage de reussite sur la segmentation
		reussi = 0
		dist = np.array([[np.inf for u in range(len(maxi))] for v in range(len(veritesTerrain)) ])
		for i, val1 in enumerate(maxi):
			for j, val2 in enumerate(veritesTerrain):
				temp = abs(val1-val2)
				if temp < 8:
					dist[j][i] = temp
				
		while dist.min() != np.inf and reussi < len(veritesTerrain):
			reussi += 1
			tp = dist.argmin()
		#	print  dist.min() , ", ", tp
			x = tp/len(maxi)
			y = tp%len(maxi)
			for i in range(len(maxi)) : 
				dist[x][i] = np.inf
			for j in range(len(veritesTerrain)) : 
				dist[j][y] = np.inf
		#print reussi		
		return reussi

	#### Calcul la precision	
	def precision(self, maxi, reussi) : # [], []
		return float(reussi)/len(maxi)


	#### Calcul le rappel
	def rappel(self, reussi, veritesTerrain) : # [], []
		return float(reussi)/len(veritesTerrain)
	
	#### Evalue les resultats par rapport aux verites terrain 
	def evaluation(self, veritesTerrain, maxi) : # [], [] -> Verites terrain donnees en secondes 
		reussi = self.reussis(maxi, veritesTerrain)
		prec = self.precision(maxi, reussi)
		rapp = self.rappel(reussi, veritesTerrain)
		return prec, rapp

	
	#### Boucle de detection 
	
	def boucle_detect_eval(self, range_order, range_seuil, vT) :
		for i,val in enumerate(vT) :
			vT[i] = round(val*self.fps) 
 		resultat = [[0.0 for u in range(range_order)] for v in range(range_seuil)]
		for i in range(range_order) :
			for j in range(range_seuil):
				self.order = 5+i*2		
				self.seuil =  0.01 + j *0.01
				maxi = self.detect()
				prec, rapp = self.evaluation(vT, maxi)
				fmesure = 0.0
				tp = rapp + prec
				if tp != 0.0 : 
					fmesure = 2*prec*rapp/(tp)

				resultat[j][i] = (prec,rapp,fmesure)	
				print "[%d, %d] = %d, %d" % (i, j, rapp, prec, )
		return resultat

	#### Permet de save en image le graphique obtenu
	def save(self, tailleVid, fenetreIMG, nomIMG, affichage) : #int, int, str, bool
		incrDist = 0
		incrMax = 0
		incrX = 0
		for i in range(int(tailleVid/fenetreIMG)):
			tempMax = []
			tempX = []
			tempMoy = []
			while incrMax < len(maxi) and maxi[incrMax] > i*fenetreIMG and maxi[incrMax] <= (i+1)*fenetreIMG:
				tempMax.append(maxi[incrMax])
				incrMax +=1
			while incrX < len(veritesTerrain) and veritesTerrain[incrX] > i*fenetreIMG and x[incrX] <= (i+1)*fenetreIMG:
				tempX.append(veritesTerrain[incrX])
				incrX +=1
			# Ajoute la courbes des distances
			tempIndex = [fenetreIMG*i+u for u in range(fenetreIMG)]
			plt.plot(tempIndex,self.distMoy[(i*fenetreIMG):((i+1)*fenetreIMG)], 'g') # x, y, couleur
			# Ajoute les barres pour visualiser les verites terrain
			if len(tempMax) != 0 :
				plt.hist(tempMax, tailleVid/2, facecolor='b')
			hauteur = [1 for u in range(len(tempX))]
			plt.plot(tempX, hauteur, 'ro')
			# Parametres du graphe
			plt.title('TBBT Dist') #titre
			plt.ylim([0,3]) #taille des ordonnees
			plt.xlabel('Frames')
			plt.ylabel('Distances')
			if affichage:
				plt.ion() # permet de continuer a utiliser le terminal
				plt.show() # Affiche
			nom = "%d%d" % (nomIMG,i, )  
		 
			plt.savefig(nom)
			plt.close()

