# 100daysofCVND 30daysofudacity
Updating and tracking progress for [#30daysofudacity](https://sites.google.com/udacity.com/udacity-community-guide/community/30daysofudacity) and #100daysofCVND while learning Udacity [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) as part of [Secure and Private AI Scholarship Challenge 2019 (Facebook and Udacity)](https://sites.google.com/udacity.com/secureprivateai-phase-2/home?authuser=0)


## Day 10/30(October 09 2019):

> 1. Learnt about Haar Cascades for object detection and used pretrained face-detector architecture
	to detect faces using OpenCV 
	
	- Utilizes many positive and negative labeled images to extract Haar features which detects different lines
	and shapes
	- In the next step of Haar Cascade, different regions of the image are searched for matching by a cascade of 
	the extracted Haar features( they are tried on succession ) removing searched part of the image 
	if the classification outcome is negative on that part for a feature effectively reducing the image space 
	to search for faces.

<img src="images/10.nitc1.jpg">
	
> 2. Learnt about Algorithmic Bias 

	- Bias occurs when the training set isn't a good representative of the general population the model is 
	supposed to make predictions on.
	- Bias that creeps into our models can be a huge problem the intensity of which varies by the use case. 
	- Use cases like probability of committing crime can cause problems if model is sensitive to different
	face shapes and ethnic groups.
	
> 3. Implemented a Real-time face detector for both images and video based on this [article](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)

	- OpenCV has a built-in deep neural network module which has pretrained model for face detection.

			
## Day 9/30(October 08, 2019):

> 1. Completed [AI programming for Robotics localization](https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373) exercises 

	* Programmed basic localization
		- Localisation primarily involves starting out with an INITIAL BELIEF of the robots surroundings(probability of its position)
		- You sense objects in the environment to increase knowledge of the robots location
		- When the robot moves, the location of the robot becomes more uncertain(assuming not exact motion)
	
> 2. Started off with [Project1: Facial Keypoint Detection](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

	* Completed loading and visualizing data, overrided Pytorch Dataset class.
	for taking in the dataset. 
		- __init__ : is run when the class is instantiated
		- __call__ : required to call a class instance
		- __len__ : to use the len() function
		- __getitem__: to index the class instance 
	
> 3. [Learnt](https://realpython.com/python-coding-interview-tips/) some Python built-in library tricks 

	* get(), setdefault(), defaultdict(), Counter(), f-strings

> 4. [Dive into Deep Learning](https://www.d2l.ai/index.html) book - Introduction

	* [Reinforcement Learning](https://www.d2l.ai/chapter_introduction/intro.html#reinforcement-learning)
		- RL differs from supervised and unsupervised learning in the sense that the latter types of learning 
			doesn't affect/consider the environment the data was collected from 
		- RL as:
			1. Markov Decision Process - when the environment in which learning occurs is fully observed 
				eg: Playing chess(environment fully observed), Self Driving Car(environment-part of roads only in the range of sensors are observed)
			2. Contextual Bandit Problem - when your actions doesn't affect the subsequent state of the environment but you utilize info from the environment(context)
			3. Multi-armed Bandit Problem - wherein you take actions and try to find out which actions maximise the reward but doesn't get any information from the environment.
			
> 5. DSC HIT GCP challenge - Quest 2: [Intro to ML - Image Processing](https://www.qwiklabs.com/quests/85)

	* Completed [APIs Explorer: Qwik Start](https://www.qwiklabs.com/focuses/2457?parent=catalog) hand-on lab
			
## Day 8/30(October 07, 2019)::

> 1.Learnt about Hough transform for circles

> 2.Started learning [Dive into Deep Learning book](https://www.d2l.ai/index.html) - Introduction

> 3.Completed [WorldQuant University](https://wqu.org/programs/data-science) OOPS mini-project - coded k-means from scratch


## Day 7/30(October 06, 2019):

> 1. Learnt about Hough transform and used OpenCV to detect edges using Hough Transform
      <img src="images/7.hough-line.jpg">
      

## Day 6/30(October 05, 2019):

> 1. Discussed computer vision and tips to make good progress in the nanodegree in the first Computer Vision weekly meetup
      <img src="images/6.cvnd_meetup.png">
 
> 2. Completed Baseline: Data, ML, AI Qwiklabs quest as part of GCP #gcpchallenge
      <img src="images/6.qwiklabs-baseline-data-ml.png">

> 3. Watched MIT Self Driving Car State of the Art lecture


## Day 5/30(October 04, 2019):

**1. CVND**:

   * Learnt Canny edge detection and techniques involved: non-max suppression(thinning) and hysteresis thresholding(edge completion)
	<img src="images/5.canny_brain.jpg">


## Day 4/30(October 03, 2019):

> Digged deep in to fourier transforms and how they work? 
	Seems like there's a lot of applications to it. No wonder why Professor Gilbert Strang said that FFT is the most important numerical algorithm of our lifetime. There's still some things about it I don't really understand. But that's okay there's 2mrw.

   
## Day 3/30(October 02, 2019):

> **1. CVND**:

   * Learnt about low pass filters. Used OpenCV GaussianBlur function to blur brain image for reducing noise. Compared edge detection(sobel filter) on the blurred image and original image 
      <img src="images/3.Gaussian-brain.jpg" width="640">
   * Used fourier transform to visualize frequency spectrum of images and their filtered version
      <img src="images/3.freq-spectrum-ori-vs-blur.jpg" width="640">


## Day 2/100 (October 01, 2019):

> **1. CVND**:

   * Using Fourier transforms in Numpy to find frequency distribution of images ie, the variation of intensity in images. Fourier transform TRANSFORMS images in the x-y spatial space to the frequency space
      <img src="images/2.Fourier Transform.jpg" width="640">
   * High pass filters and finding edges of images using Sobel operator in OpenCV
      <img src="images/2. edge-sobel.jpg" width="640">


## Day 1/100 (September 30, 2019):

> I do herby solemnly ...

   <img src="images/1.30daysofudacity-pledge.png" width="640">

> **1. CVND**:

   * Change image background by basic image processing techniques using OpenCV
      <img src="images/1.nabhan1-space.jpg">
   
   * Day/Night image classifier on 200 images from the AMOS(Archive of Many Outdoor Scenes) dataset with an accuracy of 0.9375 by only manual feature extraction making use of HSV color-space
      <img src="images/1.day-night-classification.jpg">



<!--
## Day 18/60:

1. Lesson 5 - [Introduction to deep learning with pytorch](https://classroom.udacity.com/courses/ud188)
  * Learnt about stride, padding and pooling layers
  * How to implement CNN in Pytorch
  * [Capsule](https://cezannec.github.io/Capsule_Networks/) [Networks](https://classroom.udacity.com/courses/ud188/lessons/b1e148af-0beb-464e-a389-9ae293cb1dcd/concepts/8caa6477-176c-49eb-b09e-c48f373c9f68)

2. Went through a [cou](https://aircloak.com/explaining-differential-privacy/)[ple](https://accuracyandprivacy.substack.com/) of articles on Differential Privacy
  * [Takeaway](https://accuracyandprivacy.substack.com/about): US Census 2020 will be making use of DP

3. Basic analysis of [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) - [kaggle kernel](https://www.kaggle.com/nabhanpv/a-structured-approach-to-solving-the-unsinkable)

4. Came 2nd(/~15) in [Live Kahoot Quiz](https://create.kahoot.it/share/local-and-global-differential-privacy-secure-ai-challenge/8b12f5dc-de5e-4f74-8c65-6709c24c8a88) on **Local and Global Differential Privacy**

5. 50/80 in SPARC group quiz on CNN 

6. Meetup with project team for [Kaggle](http://kaggle.com) *[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/team)* for classifying *Diabetic Retinopathy*
  * Shared progress of CNN lectures in [Intro to DL with Pytorch course](https://classroom.udacity.com/courses/ud188)
  * Registered as a team for the competition for the meetup
  * Decided to use github [repo](https://github.com/nabhanabdulla/kaggle-aptos19-challenge) for collaboration

<div class="slack-update" id="slack-update-18" float="left">

  <img src="https://drive.google.com/uc?export=view&id=1Lz7--nsoEnMJETKVFx5NUqobj0vxl6Jc" alt="Slack update day#18" class="slack-update" id="slack-update-18-0" width="45%">

  <img src="https://drive.google.com/uc?export=view&id=1XGKoXuxpLIfzO6DL6BtR6NPNqAx4KCoq" alt="Slack update day#18" class="slack-update" id="slack-update-18-1" width="45%">

</div>

  </div>
</div>
-->
