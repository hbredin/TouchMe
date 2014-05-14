

import cv2
import cv
import numpy as np
from scipy import signal as sig
import Evaluation
import time 

def segmentation_plan(adresseVid):
	print "debut"
	print time.localtime()
	#Avec les parametres pour lintersection
	cont = Contours(1.6, 7, adresseVid)
	print "contours cree"
	print time.localtime()
	hist = Histogrammes(4.1, 5, adresseVid)
	print "histo cree"
	print time.localtime()
	hist.calcul_dist()
	print "ok dist hist"
	print time.localtime()
	cont.calcul_dist()
	print "ok dist cont"
	print time.localtime()
	hypo_contours = cont.detect()
	print "ok detect cont"
	print time.localtime()
	hypo_histogrammes = hist.detect()
	print "ok detect hist"
	print time.localtime()


	# Intersection des deux pour obtenir le res final
	detect = 0
	dist = np.array([[np.inf for u in range(len(hypo_contours))] for v in range(len(hypo_histogrammes)) ])
	temporaire = np.array([0.0 for v in range(len(hypo_contours))])

	for i, val1 in enumerate(hypo_contours):
		for j, val2 in enumerate(hypo_histogrammes):
			temp = abs(val1-val2)
			#print temp
			if temp < 8:
				dist[j][i] = temp

	while dist.min() != np.inf :

		tp = dist.argmin()
		x = tp/len(hypo_contours)
		y = tp%len(hypo_contours)
		for i in range(len(hypo_contours)) : 
			dist[x][i] = np.inf
		for j in range(len(hypo_histogrammes)) : 
			dist[j][y] = np.inf
		temporaire[detect] = np.around((hypo_histogrammes[x]+hypo_contours[y])/2, 1)	
		detect += 1
	temporaire.sort() 
	
	return np.take(temporaire, temporaire.nonzero()) #[] de l'intersection des resultats

class Contours :

	#### Constructeur
	def __init__(self, alph, orde, adresseVid) :
		self.order = orde
		self.alpha = alph
		self.med = []
		self.distMoy = []
		self.capture = cv2.VideoCapture(adresseVid)
		self.taille = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

	#### Calcul la distance entre les images a partir des contours
	def calcul_dist(self) : # Return : distanceMoy, med
		temp = [0.0 for u in range(2) ]
		tempdil = [0.0 for u in range(2) ]
		
		ret , frame_o = self.capture.read()
		# Lecture de l image puis diminution de la taille
		newx, newy = frame_o.shape[1]/3,frame_o.shape[0]/3
		frame = cv2.resize(frame_o,(newx,newy))

		# Extraction des contours puis dilatation
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
		edge = cv2.Canny(frame,100,200)
		dilatation = np.array(cv2.dilate(edge, element), dtype=bool)
		
		temp[0] = np.array(edge, dtype=bool)
		tempdil[0] = dilatation
		resultin = np.array([0.0 for u in range(self.taille)])
		resultout = np.array([0.0 for u in range(self.taille)])
		for v in range (self.taille-1):
			ret , frame_o = self.capture.read()
			# Lecture de l image puis diminution de la taille
			newx, newy = frame_o.shape[1]/3,frame_o.shape[0]/3
			frame = cv2.resize(frame_o,(newx,newy))

			# Extraction des contours puis dilatation
			temp[1] = cv2.Canny(frame,100,200)
			tempdil[1] = np.array(cv2.dilate(temp[1], element), dtype=bool)
			temp[1] = np.array(temp[1], dtype=bool)
			tmpf = np.sum(temp[0])
			if tmpf != 0 :
				resultout[v] = 1.0 - (float(np.sum(temp[0]*tempdil[1])) / float(tmpf))
				resultin[v] = 1.0 - (float(np.sum(tempdil[0]*temp[1])) / float(tmpf) )
			if tmpf == 0 :
				resultout[v] = 0.0
				resultin[v] = 0.0
			temp[0] = temp[1]
			tempdil[0] = tempdil[1]
			#print v, tmpf, resultin[v], resultout[v]

		result = np.maximum(resultin,resultout)
		
		#Calcul de la mediane
		tmp = [0.0 for u in range(len(result))]
		fenetreMed = 30 # Regarde a +/- 45 autour pour faire la mediane
		for i in range(len(result)) :
			if i-fenetreMed < 0 :
				tmp[i]= np.median(result[0:i+fenetreMed])
			if i+fenetreMed >= len(result) :
				tmp[i]= np.median(result[i-fenetreMed:len(result)-1])
			if i-fenetreMed >= 0 and i-fenetreMed < len(result):
				tmp[i]= np.median(result[i-fenetreMed:i+fenetreMed])
		""" # Enregistre les resultats dans des fichiers pour ne pas avoir a le faire plusieurs fois		
		nom = "%s_med_edge.pkl" % (self.nomVideo, )
		g = open(nom,"wb")
		pk.dump(tmp, g)
		g.close()
		nom = "%s_edge.pkl" % (self.nomVideo, )
		g = open(nom,"wb")
		pk.dump(result, g)
		g.close()"""
		self.med = tmp
		self.distMoy = result
		return self.distMoy, self.med # [], []
		
	#### Retourne un array d'hypotheses de coupures de plan
	def detect (self):
		# Recupere les maxima locaux de la courbe -> return les index
		hypotheses = sig.argrelmax(np.array(self.distMoy), order = self.order)
		# recupere les valeurs correspondants des hypotheses
		valDist = np.take(self.distMoy, hypotheses)
		#  Enleve les valeurs inferieures au seuil
		temp = []
		incr = 0
		while incr < valDist.size :
			if valDist.item(incr) > self.alpha*self.med[hypotheses[0][incr]] :
				temp.append(hypotheses[0][incr])
			incr+=1
		hypotheses = np.array(temp)
            	return hypotheses # np.array

	
	#### Boucle de detection pour obtenir les meilleurs parametres
	def boucle_detect_eval(self, range_order, range_seuil) :
		if self.distMoy == [] :
			self.calcul_dist()
	
 		resultat = [[0.0 for u in range(range_order)] for v in range(range_seuil)]
		for i in range(range_order) :
			for j in range(range_seuil):
				self.order = 4 + i		
				self.alpha =  1 + 0.1 * j # en fonction de j plus tard
				self.detect()
				resultat[j][i] =  Evaluation.evalu()
		return resultat


class Histogrammes :

	#### Constructeur
	def __init__(self, alph, orde, adresseVid) :
		self.order = orde
		self.alpha = alph
		self.med = []
		self.distMoy = []
		self.capture = cv2.VideoCapture(adresseVid)
		self.taille = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

	#### Calcule la distance entre les images a partir des histogrammes (rgb)
	def calcul_dist(self):
		pasComp = 10
		t=[0]*pasComp # Initialise le tableau des dix images que nous allons traiter (Le pas de comparaison : cad on compare a +4/-5 images de l'image ou on est)
		for s in range(pasComp):
		    val , temp = self.capture.read()
		    if val : t[s] = temp
		    else : t[s] = None

		dis = [[0 for u in range(self.taille) ] for v in range(3) ] # 0 = b, 1 = g, 2 = r
		color = ('b','g','r')
		i = pasComp/2 
		while t[pasComp-1] is not None and i < self.taille: #Tant qu'il n'y a pas de True dans la table	
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

		distMoy = [0 for u in range(self.taille)]
		for i, val in enumerate(dis[2]) :
			distMoy[i] = val
		for i, val in enumerate(dis[1]) :
			distMoy[i] += val
		for i, val in enumerate(dis[0]) :
			distMoy[i] += val
			distMoy[i] = distMoy[i]/3
		self.distMoy = distMoy

		#Calcul de la mediane
		tmp = [0.0 for u in range(len(distMoy))]
		for i in range(len(distMoy)) :
			if i-45 < 0 :
				tmp[i]= np.median(distMoy[0:i+45])
			if i+45 >= len(distMoy) :
				tmp[i]= np.median(distMoy[i-45:len(distMoy)-1])
			if i-45 >= 0 and i-45 < len(distMoy):
				tmp[i]= np.median(distMoy[i-45:i+45])
		"""
		nom = "%s_dist.pkl" % (self.nomVideo, )
		g = open(nom,"wb")
		pk.dump(distMoy, g)
		g.close()
		nom = "%s_med.pkl" % (self.nomVideo, )
		g = open(nom,"wb")
		pk.dump(tmp, g)
		g.close()
		"""
		self.med = tmp
		return self.distMoy, self.med

	#### Retourne un array d'hypotheses de coupures de plan
	def detect (self):
		# Recupere les maxima locaux de la courbe -> return les index
		hypotheses = sig.argrelmax(np.array(self.distMoy), order = self.order)
		# recupere les valeurs correspondants des hypotheses
		valDist = np.take(self.distMoy, hypotheses)
		#  Enleve les valeurs inferieures au seuil
		temp = []
		incr = 0
		while incr < valDist.size :
			if valDist.item(incr) > self.alpha*self.med[hypotheses[0][incr]] :
				temp.append(hypotheses[0][incr])
			incr+=1
		hypotheses = np.array(temp)
            	return hypotheses # np.array
	
	#### Boucle de detection pour obtenir les meilleurs parametres
	def boucle_detect_eval(self, range_order, range_seuil) :
		if self.distMoy == [] :
			self.calcul_dist()
	
 		resultat = [[0.0 for u in range(range_order)] for v in range(range_seuil)]
		for i in range(range_order) :
			for j in range(range_seuil):
				self.order = 4 + i		
				self.alpha =  1 + 0.1 * j # en fonction de j plus tard
				self.detect()
				prec, rapp = self.evaluation()
				fmesure = 0.0
				tp = rapp + prec
				if tp != 0.0 : 
					fmesure = 2*prec*rapp/(tp)

				resultat[j][i] = (prec,rapp,fmesure)	
				#if rapp > 0.90 and prec > 0.90: 
					#print "[%d, %f] = %f, %f       !!!!!!!!!!!!!!!" % (self.order, self.alpha, rapp, prec, )
				#print "[%d, %f] = %f, %f" % (self.order, self.alpha, rapp, prec, )
		return resultat
