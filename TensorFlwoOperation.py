#TensorFlow operations 

#1) a+b+cd
import tensorflow as tf
a = tf.placeholder(tf.int32, name="input_a")
b = tf.placeholder(tf.int32, name="input_b")
c = tf.placeholder(tf.int32, name="input_c")
d = tf.placeholder(tf.int32, name="input_d")
add_ab = tf.add(a,b,name="addAb")
mlt_cd = tf.multiply(c,d,name="multcd")
out = tf.add(add_ab,mlt_cd, name="output")
sess = tf.Session()
feed_dict = {a:2, b:4,c:3,d:2}
result = sess.run(out,feed_dict=feed_dict)
print(result)
print("({0}+{1}) + ({2}*{3}) = {4}".format(feed_dict[a], feed_dict[b],feed_dict[c],feed_dict[d], result))

#2) (a+b)(a+b)+c

a = tf.placeholder(tf.int32, name="input_a")
b = tf.placeholder(tf.int32, name="input_b")
c = tf.placeholder(tf.int32, name="input_c")
add_ab = tf.add(a,b, name="addAb")
mult_ab_ab = tf.multiply(add_ab,add_ab,name="mult_ab")
out = tf.add(mult_ab_ab,c, name="output")
sess = tf.Session()
feed_dict = {a:2,b:3,c:10}
result = sess.run(out,feed_dict=feed_dict)
print("({0}+{1})*({0}+{1}) + {2} = {3}".format(feed_dict[a], feed_dict[b],feed_dict[c],result))

#3) a+(bc)-a/d

a = tf.placeholder(tf.int32, name="input_a")
b = tf.placeholder(tf.int32, name="input_b")
c = tf.placeholder(tf.int32, name="input_c")
d = tf.placeholder(tf.int32, name="input_d")
mult_bc = tf.multiply(b,c,name="multbc")
div_ad = tf.div(a,d,name="divad")
sub_bc_ad = tf.subtract(mult_bc,div_ad,name="addmultdiv")
out = tf.add(a,sub_bc_ad, name="output")
sess = tf.Session()
feed_dict = {a:2,b:1,c:3,d:4}
result = sess.run(out,feed_dict=feed_dict)

print("{0}({1}*{2}) - ({0}/{3})  = {4}".format(feed_dict[a], feed_dict[b],feed_dict[c],feed_dict[d],result))


#4) ax²+bx+c
a = tf.placeholder(tf.int32, name="input_a")
b = tf.placeholder(tf.int32, name="input_b")
c = tf.placeholder(tf.int32, name="input_c")
x = tf.placeholder(tf.int32, name="input_x")
x2 = tf.multiply(x,x,name="xquad")
ax2 = tf.multiply(a,x2,name="ax2")
bx = tf.multiply(b,x,name="bx")
bxc = tf.add(bx,c,name="bxc")
out = tf.add(ax2,bxc,name="output")
sess = tf.Session()
feed_dict = {a:1,x:2,b:3,c:2}
result = sess.run(out,feed_dict=feed_dict)

print("({0}*{3}^2) + ({1}*{3}) + {2} = {4}".format(feed_dict[a], feed_dict[b],feed_dict[c],feed_dict[x],result))

