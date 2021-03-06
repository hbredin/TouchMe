/* Gestion globale des elements de la page */
// Objets
var vid, time, acc, posX, posY, touch, tmp, div1;

// Timer 
var vidTimer;

// Segmentation
var temp = '[10.414, 12.734, 15.134, 18.414, 24.174, 26.214, 33.454, 37.774, 40.654, 41.894, 48.374, 50.254, 57.414, 60.654, 64.334, 65.774, 68.014, 73.454, 81.934, 85.294, 88.454, 99.934, 106.014, 108.294, 109.414, 118.414, 124.894, 128.654, 132.814, 135.094, 137.094, 137.934, 138.694, 142.654, 147.374, 149.374, 151.334, 173.534, 182.174, 183.814, 185.734, 187.494, 194.494, 195.294, 196.174, 197.454, 200.414, 202.774, 205.094, 207.014, 215.494, 216.614, 218.054, 219.014, 220.254, 220.814, 221.534, 223.814, 224.534, 228.454, 230.854, 231.494, 232.014, 234.014, 235.094, 237.254, 238.894, 243.614, 246.174, 248.854, 250.574, 251.974, 256.134, 259.814, 262.174, 263.614, 269.814, 273.054, 274.934, 281.774, 285.174, 286.534, 293.894, 295.534, 297.134, 298.614, 299.414, 300.974, 305.734, 307.214, 314.174, 315.894, 323.334, 324.454, 329.894, 332.134, 334.134, 336.294, 339.814, 342.334, 343.494, 371.294, 374.774, 375.454, 376.534, 377.814, 383.494, 384.934, 387.094, 390.934, 394.174, 398.414, 401.254, 403.654, 406.494, 408.374, 409.494, 415.094, 417.014, 420.654, 424.694, 426.614, 427.494, 429.374, 430.294, 436.734, 438.614, 442.734, 443.854, 452.014, 454.494, 460.294, 461.454, 466.894, 468.454, 470.374, 471.414, 472.294, 475.054, 492.854, 493.854, 495.814, 498.254, 500.814, 502.454, 505.054, 506.574, 508.134, 509.934, 512.134, 514.054, 516.414, 517.894, 522.454, 523.814, 527.574, 534.174, 537.014, 538.454, 543.414, 544.614, 547.774, 549.614, 552.454, 554.654, 555.774, 557.094, 562.294, 563.334, 566.894, 568.614, 570.654, 574.214, 576.294, 578.174, 579.494, 581.574, 582.934, 586.654, 592.894, 594.694, 596.774, 598.374, 600.054, 602.134, 605.254, 608.294, 611.374, 613.054, 620.734, 623.334, 625.414, 627.734, 630.934, 632.734, 633.494, 634.534, 641.534, 642.374, 645.214, 647.374, 656.454, 658.374, 659.414, 672.494, 674.894, 677.894, 684.734, 690.294, 692.934, 695.254, 698.894, 700.614, 703.614, 705.254, 707.694, 712.574, 721.454, 728.254, 733.934, 737.454, 742.334, 745.374, 747.454, 749.374, 752.174, 754.654, 758.774, 765.214, 772.334, 780.094, 790.494, 791.734, 793.294, 796.934, 798.254, 801.134, 802.934, 804.854, 805.774, 806.534, 807.494, 809.134, 820.494, 821.214, 824.174, 826.574, 827.894, 830.574, 833.214, 835.374, 836.894, 838.174, 839.614, 840.494, 843.054, 844.934, 847.654, 855.014, 858.574, 863.934, 868.494, 871.974, 875.614, 882.734, 884.974, 886.254, 888.134, 890.574, 901.774, 930.654, 933.454, 940.094, 940.894, 942.974, 946.494, 952.134, 954.134, 956.134, 958.494, 961.094, 964.134, 967.014, 967.934, 978.334, 988.014, 991.854, 995.534, 1000.33, 1004.77, 1007.77, 1009.13, 1010.77, 1011.93, 1015.05, 1017.53, 1023.37, 1026.25, 1027.73, 1030.09, 1034.29, 1042.09, 1044.93, 1046.85, 1057.33, 1058.25, 1060.97, 1067.49, 1068.77, 1070.13, 1071.53, 1073.33, 1075.65, 1080.17, 1083.61, 1088.33, 1089.61, 1090.65, 1095.85, 1097.45, 1100.81, 1101.81, 1105.89, 1125.57, 1145.09, 1163.25, 1165.21, 1170.57, 1175.53, 1180.61, 1182.73, 1183.93, 1186.73, 1191.05, 1192.73, 1194.93, 1200.49, 1205.21, 1210.17, 1211.09, 1218.13, 1222.41, 1223.65, 1226.09, 1227.53, 1228.57, 1259.37, 1262.65, 1264.41, 1265.77, 1271.09, 1271.65, 1275.21, 1280.69, 1284.21, 1288.45, 1291.77]';
var segm;

var planActuel = 0;
// Pour le multitouch
var hammertime;

// Recupere tous les elements
function init() {

	/*var xmlhttp;
	if (window.XMLHttpRequest){ // Ne fonctionne pas avec IE
		xmlhttp = new XMLHttpRequest();
	}

	xmlhttp.onreadystatechange = function(){
		if (xmlhttp.readyState==4 && xmlhttp.status==200){ // Si tout s'est bien passe
			var tempTotal = xmlhttp.responseText;
			if(typeof JSON!="undefined") { // Verifie Json
								
			} else {
				alert("PROBLEME Json !");
			}
			
		}
	}
	xmlhttp.open("GET","text.txt",true);
	xmlhttp.send();*/

	if(typeof JSON!="undefined") { // Verifie Json
		segm = eval(temp);
		console.log(segm);						
	} else {
		alert("PROBLEME Json !");
	}
			

	// Canvas, context et document 
	//can = document.getElementById("can");
	//ctx = can.getContext("2d");

	// Objects : button et video 
	vid = document.getElementById("vid");
	vid.addEventListener('ended', vidEnd, false);

	// Affichages 
	time = document.getElementById("curtime");
	debug = document.getElementById("debug");
	pos = document.getElementById("position");
	tmp = document.getElementById("temps");	
	vidTimer = window.setInterval("plans()",1);
	
}

/* Gestion du multitouch */
hammertime = Hammer(vid);
console.log(hammertime);

// Gestion des differents mouvements
hammertime.on("doubletap", function(e){playVideo(); touchXY(e);}); // Play/pause
hammertime.on("pinchin", function(e){stop(); touchXY(e);}); // Stop
hammertime.on("swiperight", function(e){fwd(); touchXY(e);}); // Avance de 10 secs -> Faire plutot passer au plan suivant ?
hammertime.on("swipeleft", function(e){bwd(); touchXY(e);}); // Recule de 10 secs -> Faire plutot revenir au plan precedent ?



// Affiche la position
function touchXY(e) {
	pos.innerText = Math.round(vid.currentTime) + " : ";
	for (i = 0; i < e.gesture.touches.length; i++) {
 		x = e.gesture.touches[i].pageX;
		y = e.gesture.touches[i].pageY;
		showPos(x, y); 
	}
}

// Affiche la position et le cercle sous le doigt
function showPos(canX, canY){
  	pos.innerText = pos.innerText + " (" + canX + "," + canY + ") ";

	/* A remettre quand j'aurai enfin le canvas. 
	ctx.clearRect(0, 0, document.width, document.height);

	ctx.beginPath();
	ctx.arc(canX, canY, 10, 0, 2 * Math.PI, false);
	ctx.fillStyle = 'white';
	ctx.fill(); */
}



function curtime(){
	time.innerText = Math.round(vid.currentTime);
	if(vid.currentTime > 0 && vid.currentTime <= 1){ // Oblige car au tout debut on a pas la duree. Cest que au bout de une seconde 
		tmp.innerText = Math.round(vid.duration); // Returns the duration in seconds of the current media resource. A NaN value is returned if duration is not available, or Infinity if the media resource is streaming 
	}	
}


/* Gestion de la video */

function plans(){ 
	curtime();
	if(vid.currentTime >= (segm[planActuel] - 0.15)){
		planActuel +=1;
		vid.pause();
	}
	
}
// Play 
function playVideo() {
	if (vid.paused == true) {
		vid.play();
	}
	else {
		vid.pause();
	}
}

// Forward
function fwd(){
	planActuel += 1;
	vid.currentTime = segm[planActuel]-0.15;
}

// Backward 
function bwd(){
  	planActuel -= 1;
	vid.currentTime = segm[planActuel]-0.15;
}

// Stop 
function stop(){
	vid.currentTime = -1;
	planActuel = 0;
	vid.pause();
}

// Plus lentement 
function slower(){
	vid.playbackRate -= 0.25;
	vid.pause();
	vid.play();
}

// Plus rapide 
function faster(){
	vid.playbackRate += 0.25;
	vid.pause();
	vid.play();
}

// Evenement de fin de video 
function vidEnd() {
	playButton.value = "Play";
	vid.playbackRate = 1;
	clearTimeout(vidTimer)
}	

