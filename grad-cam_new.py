from keras.applications.vgg16 import (
	VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import math
import heapq
import PIL.Image as Image
from matplotlib import pyplot as plt
import os
from keras.models import Model


def target_category_loss(x, category_index, nb_classes):
	return tf.multiply(x, K.one_hot([category_index], nb_classes))			#multiply by element


def target_category_loss_output_shape(input_shape):
	return input_shape


def normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(path):
	img_path = sys.argv[1]
	img = image.load_img(img_path, target_size=(224, 224))
	#img.save('resize.jpg')
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	#print(x[0])
	#pic = Image.fromarray(np.uint8(x[0]))
	#pic.save('pre.jpg')
	#exit()
	return x

def deprocess_image(x):
	if np.ndim(x) > 3:
		x = np.squeeze(x)
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	if K.image_dim_ordering() == 'th':
		x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x


def _compute_gradients(tensor, var_list):
	grads = tf.gradients(tensor, var_list)	# shape=(?, 14, 14, 512)
	return [grad if grad is not None else tf.zeros_like(var)
			for var, grad in zip(var_list, grads)]

'''
def save_summary(path):
	file_name = []
	for file in sorted(os.listdir(path)):
		if file != 'summary.jpg' and file[-3:] == 'jpg':
			file_name.append(file)
	file_amount = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name != 'summary.jpg'])
	#print('file_name:',file_name)
	#print('file_amount:',file_amount)
	for i in range(file_amount):
		img = cv2.imread(path + file_name[i])
		plt.subplot(math.ceil(math.sqrt(file_amount)), math.ceil(math.sqrt(file_amount)) , i + 1)
		plt.imshow(img)
		plt.title(file_name[i], fontsize=5)
		plt.axis('off')
	plt.savefig(path+'summary.jpg')
'''
def Compute(num):
	a = math.ceil(math.sqrt(num))
	for i in range(a,0,-1):
		if num % i == 0:
			IMAGE_ROW = i
			IMAGE_COLUMN = int(num/i)
			return IMAGE_ROW,IMAGE_COLUMN
	return False,False

def save_summary(path,IMAGE_SIZE):
	file_name = []
	for file in sorted(os.listdir(path)):
		if file != 'summary.jpg' and file[-3:] == 'jpg':
			file_name.append(file)
	file_amount = len(file_name)
	#file_amount = len(
		#[name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name != 'summary.jpg'])
	IMAGE_ROW,IMAGE_COLUMN= Compute(file_amount)
	summary = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
	for y in range(1, IMAGE_ROW + 1):
		for x in range(1, IMAGE_COLUMN + 1):
			from_image = Image.open(path + file_name[IMAGE_COLUMN * (y - 1) + x - 1])
			summary.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
	summary.save(path+'summary.jpg')

def grad_cam(input_model, image, category_index, layer_name, effect,path):#grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
	nb_classes = 1000
	target_layer = lambda x: target_category_loss(x, category_index, nb_classes)					#该类得分的梯度
	x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)	#x即为最后的lambda层

	model = Model(inputs=input_model.input, outputs=x)

	loss = K.sum(model.output)														#最后的lambda层的输出(?,1000)

	conv_output = [l for l in model.layers if l.name is layer_name][0].output		#最后一个卷积层的输出(?,14,14,512)
	grads = normalize(_compute_gradients(loss, [conv_output])[0])					#shape=(?,14,14,512)	求导
	gradient_function = K.function([model.input], [conv_output, grads])

	output, grads_val = gradient_function([image])
	output, grads_val = output[0, :], grads_val[0, :, :, :]

	#f = open("log.txt","a")
	#f.write("output:"+str(np.shape(output))+"grad_val"+str(np.shape(grads_val)))
	#print(layer_name+"output.shape=",np.shape(output))											#output.shape=(14,14,512)
	#print(layer_name+"grads_val.shape=",np.shape(grads_val))									#grads.shape=(?,14,14,512)

	weights = np.mean(grads_val, axis=(0, 1))										#第 k 个特征图对应类别 c 的权重

	#f.write("weights:"+str(np.shape(weights))+"\n")
	#f.close()
	#print("weight.shape=:",np.shape(weights))										#weights.shape=(512,)
	'''get the original output of each kernel'''
	path = path+'/'+layer_name+'/'
	if not os.path.exists(path):
		os.makedirs(path)
	for i,w in enumerate(weights):
		#if layer_name == 'block5_conv3':
			#f = open("log.txt","a")
			#f.write("图"+str(i)+"对应的权重"+str(w)+'\n')
			#f.close()
	#output[:,:,i] = np.float32(output[:,:,i])
		#tmp = np.ones(output.shape[0:2],dtype=np.float32)
		tmp = cv2.resize(output[:,:,i],(224,224))+1
		#tmp = cv2.applyColorMap(np.uint8(255 * tmp/np.max(tmp)), cv2.COLORMAP_JET)
		#tmp = np.float32(tmp)
			#gg = image[0,:]
			#gg -= np.min(gg)
			#gg = np.minimum(gg, 255)
			#gg = cv2.resize(gg,np.shape(tmp)[0:2])
		tmp = np.float32(tmp)# + np.float32(gg)
		tmp = np.uint8(255*tmp/np.max(tmp))
		if i < 10:
			name = '00'+str(i)
		elif 9 < i < 100:
			name = '0'+str(i)
		else:
			name = str(i)
		cv2.imwrite(path + name + '.jpg',tmp)
	#save_summary(path,np.shape(output)[0])



	cam = np.ones(output.shape[0: 2], dtype=np.float32)
	x = []
	y =[]
	f = open("log.txt","a")
	f.write("------------layer:"+layer_name+"-------------")
	f.write("\n")
	for i, w in enumerate(weights):
		cam += w * output[:, :, i]													#αk*Ak
		x.append(i)
		y.append(w)
		f.write("kernel:"+str(i))
		f.write("weight:"+str(w))
		f.write("\n")
	f.close()
	#plt.scatter(x,y,s=np.pi,c=1,alpha=0.5)
	plt.plot(x,y,'ro')
	plt.show()
	plt.savefig(path + 'weights.jpg')

	#cv2.imwrite('origin_image.jpg',cam)
	#cv2.imwrite('origin_image_constraint.jpg',cam*255/np.max(cam))

	cam = cv2.resize(cam, (224, 224))
	if effect is True:
		cam = np.maximum(cam, 0)  # relu
	elif effect is False:
		cam = np.minimum(cam, 0)
		cam = -cam
	else:
		print('effect input error!')
		exit()
	heatmap = cam / np.max(cam)

	# cv2.imwrite('image.jpg',255 * cam / np.max(cam))

	# Return to BGR [0..255] from the preprocessed image
	# print("image_shape=",np.shape(image))
	image = image[0, :]
	# image = -image
	image -= np.min(image)
	image = np.minimum(image, 255)

	cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # color
	#image=np.resize(image,cam.shape)
	cam = np.float32(cam) + np.float32(image)
	cam = 255 * cam / np.max(cam)
	return np.uint8(cam), heatmap




argv2 = ['VGG16','VGG19']
if sys.argv[2] not in argv2:
	print('Input format error')
	exit(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''prediction'''
preprocessed_input = load_image(sys.argv[1])
model = VGG16(weights='imagenet')
predictions = model.predict(preprocessed_input)
print(model.summary())

predict_name = []
print('Predicted class:')
for i in range(10):
	top = decode_predictions(predictions,top=10)[0][i]
	predict_name.append(top[1])
	print('%s (%s) with probability %.3f' % (top[1], top[0], top[2]))

'''predicted_class for index'''
predictions = list(predictions[0])
predicted_class = []
for i in range(10):
	index = np.argmax(predictions)
	predicted_class.append(index)
	#predicted_class = np.argmax(predictions)
	predictions[index] = 0
#print(predicted_class)

print(predicted_class)

'''target layer name'''
layer_name = list([layer.name for layer in model.layers])
for i in range(4):
	layer_name.pop()
layer_name.remove('input_1')
print(layer_name)
j=0
'''get the layer_output of each class'''
#for j in range(4):											#len(predicted_class)
if j==0:
	path = './result/' + sys.argv[2] + '/' + sys.argv[1][11:] + '/' + str(predicted_class[j]) + '/'
	if not os.path.exists(path):
		os.makedirs(path)

	for i in range(len(layer_name)):
		cam, heatmap = grad_cam(model, preprocessed_input, predicted_class[j], layer_name[i],True,path)
		cv2.imwrite(path + layer_name[i] + '.jpg', cam)
	save_summary(path,224)

'''get the negative output'''
path = './result/' + sys.argv[2] + '/' + sys.argv[1][11:] + '/' + str(predicted_class[0]) + '_neg' + '/'
if not os.path.exists(path):
	os.makedirs(path)
for i in range(len(layer_name)):
	cam, heatmap = grad_cam(model, preprocessed_input, predicted_class[0], layer_name[i], False,path)
	cv2.imwrite(path + layer_name[i] + '.jpg', cam)
save_summary(path,224)
