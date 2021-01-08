import matplotlib.pyplot as plt
import numpy as np
from data_util import read_sequential_target
from config import TrainConfig
import math


train_conf = TrainConfig()
train_conf.set_conf("../train/train_conf.txt")
B_fw, _, B_bin, B_len = read_sequential_target(train_conf.B_dir)
B_fw_u, _, B_bin_u, B_len_u = read_sequential_target(train_conf.B_dir_test)
min = B_fw.min()
max = B_fw.max()
min_u = B_fw_u.min()
max_u = B_fw_u.max()
B_fw = 2 * ((B_fw - B_fw.min())/(B_fw.max()-B_fw.min())) - 1
B_fw_u = 2 * ((B_fw_u - B_fw_u.min()) / (B_fw_u.max() - B_fw_u.min())) - 1
B_fw = B_fw.transpose((1,0,2))
B_fw_u = B_fw_u.transpose((1,0,2))
B_bin = B_bin.transpose((1,0,2))
B_bin_u = B_bin_u.transpose((1,0,2))
action1 = B_fw[0]
action2 = B_fw[1]
action3 = B_fw_u[0]
action4 = B_fw[2]
action5 = B_fw[3]
action6 = B_fw[4]
fast_action1 = B_fw[27]
fast_action2 = B_fw[28]
fast_action3 = B_fw_u[9]
fast_action4 = B_fw[29]
fast_action5 = B_fw_u[10]
fast_action6 = B_fw_u[11]
x1 = np.arange(B_len[0])

predict1 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000000.txt')
predict1 = 2 * ((predict1 - min)/(max-min)) - 1
predict2 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000001.txt')
predict2 = 2 * ((predict2 - min_u) / (max_u - min_u)) - 1
predict3 = np.loadtxt('../train/generation/prediction/behavior_test/201207/target000002.txt')
predict3 = 2 * ((predict3 - min)/(max-min)) - 1
predict4 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000003.txt')
predict4 = 2 * ((predict4 - min)/(max-min)) - 1
predict5 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000004.txt')
predict5 = 2 * ((predict5 - min)/(max-min)) - 1
predict6 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000005.txt')
predict6 = 2 * ((predict6 - min)/(max-min)) - 1

fast_predict1 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000036.txt')
fast_predict1 = 2 * ((fast_predict1 - min)/(max-min)) - 1
fast_predict2 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000037.txt')
fast_predict2 = 2 * ((fast_predict2 - min_u) / (max_u - min_u)) - 1
fast_predict3 = np.loadtxt('../train/generation/prediction/behavior_test/201207/target000038.txt')
fast_predict3 = 2 * ((fast_predict3 - min)/(max-min)) - 1
fast_predict4 = np.loadtxt('../train/generation/prediction/behavior_train/201207/target000039.txt')
fast_predict4 = 2 * ((fast_predict4 - min)/(max-min)) - 1
fast_predict5 = np.loadtxt('../train/generation/prediction/behavior_test/201207/target000040.txt')
fast_predict5 = 2 * ((fast_predict5 - min)/(max-min)) - 1
fast_predict6 = np.loadtxt('../train/generation/prediction/behavior_test/201207/target000041.txt')
fast_predict6 = 2 * ((fast_predict6 - min)/(max-min)) - 1

predict1 = np.insert(predict1, 0, action1[0], axis=0)
predict2 = np.insert(predict2, 0, action2[0], axis=0)
predict3 = np.insert(predict3, 0, action3[0], axis=0)
predict4 = np.insert(predict4, 0, action4[0], axis=0)
predict5 = np.insert(predict5, 0, action5[0], axis=0)
predict6 = np.insert(predict6, 0, action6[0], axis=0)

fast_predict1 = np.insert(fast_predict1, 0, fast_action1[0], axis=0)
fast_predict2 = np.insert(fast_predict2, 0, fast_action2[0], axis=0)
fast_predict3 = np.insert(fast_predict3, 0, fast_action3[0], axis=0)
fast_predict4 = np.insert(fast_predict4, 0, fast_action4[0], axis=0)
fast_predict5 = np.insert(fast_predict5, 0, fast_action5[0], axis=0)
fast_predict6 = np.insert(fast_predict6, 0, fast_action6[0], axis=0)

# Plot for the first action
plt.figure()
plt.subplot(231)
plt.plot(np.arange(B_len[0]), action1[:B_len[0], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[0]), predict1[:B_len[0], 0], 'b--')
plt.plot(np.arange(B_len[0]), action1[:B_len[0], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[0]), predict1[:B_len[0], 1], '--', color='orange')
plt.plot(np.arange(B_len[0]), action1[:B_len[0], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[0]), predict1[:B_len[0], 2], 'g--')
plt.plot(np.arange(B_len[0]), action1[:B_len[0], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[0]), predict1[:B_len[0], 3], 'r--')
plt.plot(np.arange(B_len[0]), action1[:B_len[0], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[0]), predict1[:B_len[0], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
plt.ylabel('Joint angles')
plt.title('PUSH-L-SLOW')
plt.legend()
#plt.show()

# Plot for the second action
plt.subplot(232)
plt.plot(np.arange(B_len_u[0]), action2[:B_len_u[0], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len_u[0]), predict2[:B_len_u[0], 0], 'b--')
plt.plot(np.arange(B_len_u[0]), action2[:B_len_u[0], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len_u[0]), predict2[:B_len_u[0], 1], '--', color='orange')
plt.plot(np.arange(B_len_u[0]), action2[:B_len_u[0], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len_u[0]), predict2[:B_len_u[0], 2], 'g--')
plt.plot(np.arange(B_len_u[0]), action2[:B_len_u[0], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len_u[0]), predict2[:B_len_u[0], 3], 'r--')
plt.plot(np.arange(B_len_u[0]), action2[:B_len_u[0], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len_u[0]), predict2[:B_len_u[0], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('PULL-L-SLOW')
plt.legend()


#
# # Plot for the third action
plt.subplot(233)
plt.plot(np.arange(B_len[1]), action3[:B_len[1], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[1]), predict3[:B_len[1], 0], 'b--')
plt.plot(np.arange(B_len[1]), action3[:B_len[1], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[1]), predict3[:B_len[1], 1], '--', color='orange')
plt.plot(np.arange(B_len[1]), action3[:B_len[1], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[1]), predict3[:B_len[1], 2], 'g--')
plt.plot(np.arange(B_len[1]), action3[:B_len[1], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[1]), predict3[:B_len[1], 3], 'r--')
plt.plot(np.arange(B_len[1]), action3[:B_len[1], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[1]), predict3[:B_len[1], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('SLIDE-L-SLOW')
plt.legend()
#
# # Plot for the fourth action
plt.subplot(234)
plt.plot(np.arange(B_len[2]), action4[:B_len[2], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[2]), predict4[:B_len[2], 0], 'b--')
plt.plot(np.arange(B_len[2]), action4[:B_len[2], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[2]), predict4[:B_len[2], 1], '--', color='orange')
plt.plot(np.arange(B_len[2]), action4[:B_len[2], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[2]), predict4[:B_len[2], 2], 'g--')
plt.plot(np.arange(B_len[2]), action4[:B_len[2], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[2]), predict4[:B_len[2], 3], 'r--')
plt.plot(np.arange(B_len[2]), action4[:B_len[2], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[2]), predict4[:B_len[2], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
plt.ylabel('Joint angles')
plt.title('PUSH-R-SLOW')
plt.legend()
#
# # Plot for the fifth action
plt.subplot(235)
plt.plot(np.arange(B_len[3]), action5[:B_len[3], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[3]), predict5[:B_len[3], 0], 'b--')
plt.plot(np.arange(B_len[3]), action5[:B_len[3], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[3]), predict5[:B_len[3], 1], '--', color='orange')
plt.plot(np.arange(B_len[3]), action5[:B_len[3], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[3]), predict5[:B_len[3], 2], 'g--')
plt.plot(np.arange(B_len[3]), action5[:B_len[3], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[3]), predict5[:B_len[3], 3], 'r--')
plt.plot(np.arange(B_len[3]), action5[:B_len[3], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[3]), predict5[:B_len[3], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('PULL-R-SLOW')
plt.legend()
#
# # Plot for the sixth action
plt.subplot(236)
plt.plot(np.arange(B_len[4]), action6[:B_len[4], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[4]), predict6[:B_len[4], 0], 'b--')
plt.plot(np.arange(B_len[4]), action6[:B_len[4], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[4]), predict6[:B_len[4], 1], '--', color='orange')
plt.plot(np.arange(B_len[4]), action6[:B_len[4], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[4]), predict6[:B_len[4], 2], 'g--')
plt.plot(np.arange(B_len[4]), action6[:B_len[4], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[4]), predict6[:B_len[4], 3], 'r--')
plt.plot(np.arange(B_len[4]), action6[:B_len[4], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[4]), predict6[:B_len[4], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('SLIDE-R-SLOW')
plt.legend()
plt.show()

# Plot for the first action
plt.figure()
plt.subplot(231)
plt.plot(np.arange(B_len[28]), fast_action1[:B_len[28], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[28]), fast_predict1[:B_len[28], 0], 'b--')
plt.plot(np.arange(B_len[28]), fast_action1[:B_len[28], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[28]), fast_predict1[:B_len[28], 1], '--', color='orange')
plt.plot(np.arange(B_len[28]), fast_action1[:B_len[28], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[28]), fast_predict1[:B_len[28], 2], 'g--')
plt.plot(np.arange(B_len[28]), fast_action1[:B_len[28], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[28]), fast_predict1[:B_len[28], 3], 'r--')
plt.plot(np.arange(B_len[28]), fast_action1[:B_len[28], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[28]), fast_predict1[:B_len[28], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
plt.ylabel('Joint angles')
plt.title('PUSH-L-FAST')
plt.legend()
#plt.show()

# Plot for the second action
plt.subplot(232)
plt.plot(np.arange(B_len[28]), fast_action2[:B_len[28], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[28]), fast_predict2[:B_len[28], 0], 'b--')
plt.plot(np.arange(B_len[28]), fast_action2[:B_len[28], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[28]), fast_predict2[:B_len[28], 1], '--', color='orange')
plt.plot(np.arange(B_len[28]), fast_action2[:B_len[28], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[28]), fast_predict2[:B_len[28], 2], 'g--')
plt.plot(np.arange(B_len[28]), fast_action2[:B_len[28], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[28]), fast_predict2[:B_len[28], 3], 'r--')
plt.plot(np.arange(B_len[28]), fast_action2[:B_len[28], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[28]), fast_predict2[:B_len[28], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('PULL-L-FAST')
plt.legend()


#
# # Plot for the third action
plt.subplot(233)
plt.plot(np.arange(B_len[28]), fast_action3[:B_len[28], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[28]), fast_predict3[:B_len[28], 0], 'b--')
plt.plot(np.arange(B_len[28]), fast_action3[:B_len[28], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[28]), fast_predict3[:B_len[28], 1], '--', color='orange')
plt.plot(np.arange(B_len[28]), fast_action3[:B_len[28], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[28]), fast_predict3[:B_len[28], 2], 'g--')
plt.plot(np.arange(B_len[28]), fast_action3[:B_len[28], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[28]), fast_predict3[:B_len[28], 3], 'r--')
plt.plot(np.arange(B_len[28]), fast_action3[:B_len[28], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[28]), fast_predict3[:B_len[28], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('SLIDE-L-FAST')
plt.legend()
#
# # Plot for the fourth action
plt.subplot(234)
plt.plot(np.arange(B_len[28]), fast_action4[:B_len[28], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[28]), fast_predict4[:B_len[28], 0], 'b--')
plt.plot(np.arange(B_len[28]), fast_action4[:B_len[28], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[28]), fast_predict4[:B_len[28], 1], '--', color='orange')
plt.plot(np.arange(B_len[28]), fast_action4[:B_len[28], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[28]), fast_predict4[:B_len[28], 2], 'g--')
plt.plot(np.arange(B_len[28]), fast_action4[:B_len[28], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[28]), fast_predict4[:B_len[28], 3], 'r--')
plt.plot(np.arange(B_len[28]), fast_action4[:B_len[28], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[28]), fast_predict4[:B_len[28], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
plt.ylabel('Joint angles')
plt.title('PUSH-R-FAST')
plt.legend()
#
# # Plot for the fifth action
plt.subplot(235)
plt.plot(np.arange(B_len[28]), fast_action5[:B_len[28], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[28]), fast_predict5[:B_len[28], 0], 'b--')
plt.plot(np.arange(B_len[28]), fast_action5[:B_len[28], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[28]), fast_predict5[:B_len[28], 1], '--', color='orange')
plt.plot(np.arange(B_len[28]), fast_action5[:B_len[28], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[28]), fast_predict5[:B_len[28], 2], 'g--')
plt.plot(np.arange(B_len[28]), fast_action5[:B_len[28], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[28]), fast_predict5[:B_len[28], 3], 'r--')
plt.plot(np.arange(B_len[28]), fast_action5[:B_len[28], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[28]), fast_predict5[:B_len[28], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('PULL-R-FAST')
plt.legend()
#
# # Plot for the sixth action
plt.subplot(236)
plt.plot(np.arange(B_len[28]), fast_action6[:B_len[28], 0], 'b-', label='Joint 1')
plt.plot(np.arange(B_len[28]), fast_predict6[:B_len[28], 0], 'b--')
plt.plot(np.arange(B_len[28]), fast_action6[:B_len[28], 1], '-', color='orange', label='Joint 2')
plt.plot(np.arange(B_len[28]), fast_predict6[:B_len[28], 1], '--', color='orange')
plt.plot(np.arange(B_len[28]), fast_action6[:B_len[28], 2], 'g-', label='Joint 3')
plt.plot(np.arange(B_len[28]), fast_predict6[:B_len[28], 2], 'g--')
plt.plot(np.arange(B_len[28]), fast_action6[:B_len[28], 3], 'r-', label='Joint 4')
plt.plot(np.arange(B_len[28]), fast_predict6[:B_len[28], 3], 'r--')
plt.plot(np.arange(B_len[28]), fast_action6[:B_len[28], 4], '-', color='purple', label='Joint 5')
plt.plot(np.arange(B_len[28]), fast_predict6[:B_len[28], 4], '--', color='purple')
plt.ylim([-1.05, 1.05])
plt.xlabel('Time steps')
#plt.ylabel('Joint angles')
plt.title('SLIDE-R-FAST')
plt.legend()
plt.show()