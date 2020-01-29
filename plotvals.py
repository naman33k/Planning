import numpy as np
import matplotlib.pyplot as plt
import re

# With 0.1 wind
# ILC solution
#a = [1897967., 135280.16, 135213.1, 2669.8525, 2235.9893, 230.9986, 61.61985, 60.350395, 60.300068, 60.295013]
# IGPC Solutions
#b = [1897967., 134428.95, 87237.62, 9336.967, 398.8751, 136.4017, 60.572754, 60.327484, 60.29727, 60.29623]
#c = [1897967., 117876.945, 93900.11, 1740.1987, 1577.0483, 64.23305, 60.706566, 60.66467, 60.649284, 60.636467]
#d = [1897967., 55158.918, 38001.848, 72.22127, 62.26132, 61.12064, 60.68163, 60.59839, 60.577415, 60.55961]


# With 0.2 wind
# IGPC Solution
# e = "[DeviceArray(1.3778734e+08, dtype=float32), DeviceArray(2501159.8, dtype=float32), DeviceArray(538777.56, dtype=float32), DeviceArray(335995.5, dtype=float32), DeviceArray(307006.34, dtype=float32), DeviceArray(237745.33, dtype=float32), DeviceArray(193659.2, dtype=float32), DeviceArray(154698.06, dtype=float32), DeviceArray(124409.8, dtype=float32), DeviceArray(107471.164, dtype=float32), DeviceArray(90529.85, dtype=float32), DeviceArray(72857.06, dtype=float32), DeviceArray(63407.45, dtype=float32), DeviceArray(58003.438, dtype=float32), DeviceArray(53636.832, dtype=float32), DeviceArray(50424.03, dtype=float32), DeviceArray(47116.29, dtype=float32), DeviceArray(45237.953, dtype=float32), DeviceArray(45073.18, dtype=float32), DeviceArray(42761.57, dtype=float32), DeviceArray(40887.117, dtype=float32), DeviceArray(38249.914, dtype=float32), DeviceArray(34940.71, dtype=float32), DeviceArray(31566.57, dtype=float32), DeviceArray(28800.527, dtype=float32), DeviceArray(24161.867, dtype=float32), DeviceArray(22233.992, dtype=float32), DeviceArray(19119.758, dtype=float32), DeviceArray(17076.3, dtype=float32), DeviceArray(15357.66, dtype=float32), DeviceArray(13965.109, dtype=float32), DeviceArray(12490.787, dtype=float32), DeviceArray(11160.59, dtype=float32), DeviceArray(11036.3125, dtype=float32), DeviceArray(8886.476, dtype=float32), DeviceArray(7192.0796, dtype=float32), DeviceArray(5881.2334, dtype=float32), DeviceArray(4903.5366, dtype=float32), DeviceArray(4142.0376, dtype=float32), DeviceArray(3480.4856, dtype=float32), DeviceArray(2963.9443, dtype=float32), DeviceArray(2559.3828, dtype=float32), DeviceArray(2018.7521, dtype=float32), DeviceArray(1792.2396, dtype=float32), DeviceArray(1744.3623, dtype=float32), DeviceArray(1232.8055, dtype=float32), DeviceArray(1053.2341, dtype=float32), DeviceArray(918.4765, dtype=float32), DeviceArray(873.37396, dtype=float32), DeviceArray(844.49664, dtype=float32), DeviceArray(822.3335, dtype=float32), DeviceArray(804.7735, dtype=float32), DeviceArray(789.0145, dtype=float32), DeviceArray(773.35284, dtype=float32), DeviceArray(757.3227, dtype=float32), DeviceArray(740.0673, dtype=float32), DeviceArray(720.9356, dtype=float32), DeviceArray(698.94305, dtype=float32), DeviceArray(672.5116, dtype=float32), DeviceArray(639.2748, dtype=float32), DeviceArray(595.0775, dtype=float32), DeviceArray(533.06226, dtype=float32), DeviceArray(442.45575, dtype=float32), DeviceArray(315.54898, dtype=float32), DeviceArray(250.98587, dtype=float32), DeviceArray(148.5779, dtype=float32), DeviceArray(88.778366, dtype=float32), DeviceArray(72.88566, dtype=float32), DeviceArray(71.994354, dtype=float32), DeviceArray(71.26481, dtype=float32), DeviceArray(71.249344, dtype=float32), DeviceArray(71.246376, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32), DeviceArray(71.24616, dtype=float32)]"
# e = re.split('\(|, ',e)
# print(len(e))
# e = np.array(e)
# e = e[list(range(1, len(e), 3))]
# print(len(e))
# e = [float(v) for v in e]
# print(len(e))
# print(e)
# plt.semilogy(e)

# f="[DeviceArray(1.3778734e+08, dtype=float32), DeviceArray(28964586., dtype=float32), DeviceArray(11284699., dtype=float32), DeviceArray(3298409.5, dtype=float32), DeviceArray(3248272.8, dtype=float32), DeviceArray(2751692., dtype=float32), DeviceArray(2742162.2, dtype=float32), DeviceArray(2409707.8, dtype=float32), DeviceArray(2382586., dtype=float32), DeviceArray(2314021., dtype=float32), DeviceArray(2292981.8, dtype=float32), DeviceArray(2287670.8, dtype=float32), DeviceArray(2282602.5, dtype=float32), DeviceArray(2278010.5, dtype=float32), DeviceArray(2232198.2, dtype=float32), DeviceArray(2166271., dtype=float32), DeviceArray(2018294., dtype=float32), DeviceArray(1992428., dtype=float32), DeviceArray(1846517.2, dtype=float32), DeviceArray(1762411.4, dtype=float32), DeviceArray(1717060., dtype=float32), DeviceArray(1693744.6, dtype=float32), DeviceArray(1628920.2, dtype=float32), DeviceArray(1616905.4, dtype=float32), DeviceArray(1560759.1, dtype=float32), DeviceArray(1537697.2, dtype=float32), DeviceArray(1523880.2, dtype=float32), DeviceArray(1510589., dtype=float32), DeviceArray(1502683.2, dtype=float32), DeviceArray(1498089.8, dtype=float32), DeviceArray(1494821.9, dtype=float32), DeviceArray(1455976.2, dtype=float32), DeviceArray(1442984., dtype=float32), DeviceArray(1431583.5, dtype=float32), DeviceArray(1415775., dtype=float32), DeviceArray(1395504.2, dtype=float32), DeviceArray(1373634.9, dtype=float32), DeviceArray(1359681., dtype=float32), DeviceArray(1355482.8, dtype=float32), DeviceArray(1342215., dtype=float32), DeviceArray(1323026.6, dtype=float32), DeviceArray(1262139.5, dtype=float32), DeviceArray(1245467.2, dtype=float32), DeviceArray(1232108.5, dtype=float32), DeviceArray(1221388.2, dtype=float32), DeviceArray(1209086.8, dtype=float32), DeviceArray(1197404.6, dtype=float32), DeviceArray(1158225., dtype=float32), DeviceArray(1112121.2, dtype=float32), DeviceArray(1101394.1, dtype=float32), DeviceArray(1055071., dtype=float32), DeviceArray(1029571.8, dtype=float32), DeviceArray(1022850.94, dtype=float32), DeviceArray(1016317.06, dtype=float32), DeviceArray(1009271.6, dtype=float32), DeviceArray(1000733., dtype=float32), DeviceArray(992150.4, dtype=float32), DeviceArray(976047.5, dtype=float32), DeviceArray(959112.1, dtype=float32), DeviceArray(946641.4, dtype=float32), DeviceArray(914317.1, dtype=float32), DeviceArray(907624.9, dtype=float32), DeviceArray(890260., dtype=float32), DeviceArray(862632.8, dtype=float32), DeviceArray(853807.75, dtype=float32), DeviceArray(835526.5, dtype=float32), DeviceArray(823462.75, dtype=float32), DeviceArray(807559.2, dtype=float32), DeviceArray(796049.7, dtype=float32), DeviceArray(786160.1, dtype=float32), DeviceArray(770341.75, dtype=float32), DeviceArray(722543.1, dtype=float32), DeviceArray(703918.75, dtype=float32), DeviceArray(627392.44, dtype=float32), DeviceArray(616791.4, dtype=float32), DeviceArray(604024.94, dtype=float32), DeviceArray(592918.7, dtype=float32), DeviceArray(577450.8, dtype=float32), DeviceArray(556979.25, dtype=float32), DeviceArray(548531.25, dtype=float32), DeviceArray(539369.6, dtype=float32), DeviceArray(529444., dtype=float32), DeviceArray(508538.5, dtype=float32), DeviceArray(480109.56, dtype=float32), DeviceArray(458806.5, dtype=float32), DeviceArray(453457., dtype=float32), DeviceArray(442802.03, dtype=float32), DeviceArray(434653.4, dtype=float32), DeviceArray(428196.06, dtype=float32), DeviceArray(414830.3, dtype=float32), DeviceArray(400789.2, dtype=float32), DeviceArray(387202., dtype=float32), DeviceArray(380479.28, dtype=float32), DeviceArray(374590.34, dtype=float32), DeviceArray(357611.97, dtype=float32), DeviceArray(356065.3, dtype=float32), DeviceArray(342488.38, dtype=float32), DeviceArray(331430.38, dtype=float32), DeviceArray(314360., dtype=float32), DeviceArray(303432.1, dtype=float32), DeviceArray(301262.66, dtype=float32), DeviceArray(290909.03, dtype=float32), DeviceArray(282123.97, dtype=float32), DeviceArray(278071.78, dtype=float32), DeviceArray(273691.44, dtype=float32), DeviceArray(268885.25, dtype=float32), DeviceArray(253820.23, dtype=float32), DeviceArray(248251.06, dtype=float32), DeviceArray(242995.25, dtype=float32), DeviceArray(232932.06, dtype=float32), DeviceArray(222650.25, dtype=float32), DeviceArray(218715.25, dtype=float32), DeviceArray(205021.78, dtype=float32), DeviceArray(200415.84, dtype=float32), DeviceArray(192256.6, dtype=float32), DeviceArray(185173.33, dtype=float32), DeviceArray(180792.9, dtype=float32), DeviceArray(176473.19, dtype=float32), DeviceArray(168017., dtype=float32), DeviceArray(163991.69, dtype=float32), DeviceArray(160568.11, dtype=float32), DeviceArray(151131.03, dtype=float32), DeviceArray(138666., dtype=float32), DeviceArray(135886.83, dtype=float32), DeviceArray(130333.83, dtype=float32), DeviceArray(125920., dtype=float32), DeviceArray(118553.03, dtype=float32), DeviceArray(115106.02, dtype=float32), DeviceArray(112034.58, dtype=float32), DeviceArray(109368.36, dtype=float32), DeviceArray(106925.805, dtype=float32), DeviceArray(104512.4, dtype=float32), DeviceArray(103869.266, dtype=float32), DeviceArray(93451.734, dtype=float32), DeviceArray(92519.25, dtype=float32), DeviceArray(90995.67, dtype=float32), DeviceArray(82577.52, dtype=float32), DeviceArray(75740.82, dtype=float32), DeviceArray(73211.74, dtype=float32), DeviceArray(71567.14, dtype=float32), DeviceArray(69184.25, dtype=float32), DeviceArray(61481.83, dtype=float32), DeviceArray(58826.37, dtype=float32), DeviceArray(57128.562, dtype=float32), DeviceArray(56134.117, dtype=float32), DeviceArray(54632.555, dtype=float32), DeviceArray(52198.473, dtype=float32), DeviceArray(48618.645, dtype=float32), DeviceArray(45306.965, dtype=float32), DeviceArray(43296.375, dtype=float32), DeviceArray(37837.766, dtype=float32), DeviceArray(37187.273, dtype=float32), DeviceArray(36522.42, dtype=float32), DeviceArray(28590.715, dtype=float32), DeviceArray(27391.033, dtype=float32), DeviceArray(21472.646, dtype=float32), DeviceArray(18563.545, dtype=float32), DeviceArray(15941.457, dtype=float32), DeviceArray(13903.902, dtype=float32), DeviceArray(12599.605, dtype=float32), DeviceArray(11272.7, dtype=float32), DeviceArray(10125.377, dtype=float32), DeviceArray(8989.766, dtype=float32), DeviceArray(7868.824, dtype=float32), DeviceArray(7029.3784, dtype=float32), DeviceArray(6412.6104, dtype=float32), DeviceArray(5880.4717, dtype=float32), DeviceArray(5335.8584, dtype=float32), DeviceArray(4730.6562, dtype=float32), DeviceArray(4109.908, dtype=float32), DeviceArray(3574.7026, dtype=float32), DeviceArray(3197.1157, dtype=float32), DeviceArray(2921.455, dtype=float32), DeviceArray(2670.345, dtype=float32), DeviceArray(2622.4285, dtype=float32), DeviceArray(2580.0645, dtype=float32), DeviceArray(1762.5469, dtype=float32), DeviceArray(1637.2599, dtype=float32), DeviceArray(1608.3792, dtype=float32), DeviceArray(1280.4155, dtype=float32), DeviceArray(1103.3658, dtype=float32), DeviceArray(1057.7863, dtype=float32), DeviceArray(1037.2836, dtype=float32), DeviceArray(1018.57275, dtype=float32), DeviceArray(1002.0813, dtype=float32), DeviceArray(987.06006, dtype=float32), DeviceArray(971.43286, dtype=float32), DeviceArray(955.70135, dtype=float32), DeviceArray(939.66895, dtype=float32), DeviceArray(923.3001, dtype=float32), DeviceArray(906.5652, dtype=float32), DeviceArray(889.56714, dtype=float32), DeviceArray(872.31287, dtype=float32), DeviceArray(854.8896, dtype=float32), DeviceArray(837.3262, dtype=float32), DeviceArray(819.6369, dtype=float32), DeviceArray(801.76843, dtype=float32), DeviceArray(783.6024, dtype=float32), DeviceArray(764.9196, dtype=float32), DeviceArray(745.37775, dtype=float32), DeviceArray(724.44965, dtype=float32), DeviceArray(701.3332, dtype=float32), DeviceArray(674.76733, dtype=float32), DeviceArray(642.69684, dtype=float32), DeviceArray(601.6378, dtype=float32), DeviceArray(545.574, dtype=float32), DeviceArray(464.65448, dtype=float32), DeviceArray(347.32077, dtype=float32), DeviceArray(228.0076, dtype=float32), DeviceArray(133.97914, dtype=float32), DeviceArray(92.602844, dtype=float32), DeviceArray(73.89125, dtype=float32), DeviceArray(73.30235, dtype=float32), DeviceArray(71.33778, dtype=float32), DeviceArray(71.32229, dtype=float32), DeviceArray(71.25079, dtype=float32), DeviceArray(71.25047, dtype=float32), DeviceArray(71.249214, dtype=float32), DeviceArray(71.24793, dtype=float32), DeviceArray(71.247826, dtype=float32), DeviceArray(71.24782, dtype=float32), DeviceArray(71.24781, dtype=float32), DeviceArray(71.2478, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32)]"
# f = re.split('\(|, ',f)
# print(len(f))
# f = np.array(f)
# f = f[list(range(1, len(f), 3))]
# print(len(f))
# f = [float(v) for v in f]
# print(len(f))
# print(f)
# plt.semilogy(e)
# plt.semilogy(f)
#plt.semilogy(d)

rigpc = [2, 6, 8, 10, 13, 15, 17, 19, 21, 22, 23, 26, 28, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50, 52, 53, 55, 57, 59, 61, 63, 65, 67, 68, 71, 73, 75, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 134, 144, 154, 164, 174, 184]
vigpc = "[DeviceArray(1.3778734e+08, dtype=float32), DeviceArray(2501159.8, dtype=float32), DeviceArray(538777.56, dtype=float32), DeviceArray(335995.5, dtype=float32), DeviceArray(307006.34, dtype=float32), DeviceArray(237745.33, dtype=float32), DeviceArray(193659.2, dtype=float32), DeviceArray(154698.06, dtype=float32), DeviceArray(124409.8, dtype=float32), DeviceArray(107471.164, dtype=float32), DeviceArray(90529.85, dtype=float32), DeviceArray(72857.06, dtype=float32), DeviceArray(63407.45, dtype=float32), DeviceArray(58003.438, dtype=float32), DeviceArray(53636.832, dtype=float32), DeviceArray(50424.03, dtype=float32), DeviceArray(47116.29, dtype=float32), DeviceArray(45237.953, dtype=float32), DeviceArray(45073.18, dtype=float32), DeviceArray(42761.57, dtype=float32), DeviceArray(40887.117, dtype=float32), DeviceArray(38249.914, dtype=float32), DeviceArray(34940.71, dtype=float32), DeviceArray(31566.57, dtype=float32), DeviceArray(28800.527, dtype=float32), DeviceArray(24161.867, dtype=float32), DeviceArray(22233.992, dtype=float32), DeviceArray(19119.758, dtype=float32), DeviceArray(17076.3, dtype=float32), DeviceArray(15357.66, dtype=float32), DeviceArray(13965.109, dtype=float32), DeviceArray(12490.787, dtype=float32), DeviceArray(11160.59, dtype=float32), DeviceArray(11036.3125, dtype=float32), DeviceArray(8886.476, dtype=float32), DeviceArray(7192.0796, dtype=float32), DeviceArray(5881.2334, dtype=float32), DeviceArray(4903.5366, dtype=float32), DeviceArray(4142.0376, dtype=float32), DeviceArray(3480.4856, dtype=float32), DeviceArray(2963.9443, dtype=float32), DeviceArray(2559.3828, dtype=float32), DeviceArray(2018.7521, dtype=float32), DeviceArray(1792.2396, dtype=float32), DeviceArray(1744.3623, dtype=float32), DeviceArray(1232.8055, dtype=float32), DeviceArray(1053.2341, dtype=float32), DeviceArray(915.44165, dtype=float32), DeviceArray(872.9818, dtype=float32), DeviceArray(845.6234, dtype=float32), DeviceArray(824.27216, dtype=float32), DeviceArray(807.01196, dtype=float32), DeviceArray(791.43256, dtype=float32), DeviceArray(775.9887, dtype=float32), DeviceArray(760.1552, dtype=float32), DeviceArray(743.22595, dtype=float32), DeviceArray(724.5514, dtype=float32), DeviceArray(703.2365, dtype=float32), DeviceArray(677.8268, dtype=float32), DeviceArray(646.16425, dtype=float32), DeviceArray(604.4786, dtype=float32), DeviceArray(546.5373, dtype=float32), DeviceArray(462.25354, dtype=float32), DeviceArray(341.31363, dtype=float32), DeviceArray(277.8565, dtype=float32), DeviceArray(184.18538, dtype=float32), DeviceArray(87.44479, dtype=float32), DeviceArray(76.372444, dtype=float32), DeviceArray(74.88713, dtype=float32), DeviceArray(71.61845, dtype=float32), DeviceArray(71.301384, dtype=float32), DeviceArray(71.294815, dtype=float32), DeviceArray(71.2568, dtype=float32), DeviceArray(71.24693, dtype=float32), DeviceArray(71.24618, dtype=float32), DeviceArray(71.24618, dtype=float32), DeviceArray(71.24618, dtype=float32), DeviceArray(71.24618, dtype=float32), DeviceArray(71.24618, dtype=float32), DeviceArray(71.24618, dtype=float32)]"
vigpc = re.split('\(|, ',vigpc)
print(len(vigpc))
vigpc = np.array(vigpc)
vigpc = vigpc[list(range(1, len(vigpc), 3))]
print(len(vigpc))
vigpc = [float(v) for v in vigpc]
print(len(vigpc))
print(vigpc)
plt.semilogy(rigpc, vigpc)

rilc = [1, 2, 3, 4, 8, 11, 13, 16, 19, 21, 24, 26, 27, 28, 31, 32, 35, 37, 39, 41, 42, 46, 47, 49, 52, 54, 55, 56, 58, 60, 63, 65, 67, 69, 71, 72, 74, 76, 78, 80, 82, 84, 86, 89, 90, 92, 94, 96, 98, 99, 102, 104, 106, 107, 110, 112, 115, 117, 119, 121, 123, 125, 127, 128, 130, 133, 134, 136, 138, 139, 140, 143, 146, 147, 149, 151, 153, 156, 158, 160, 162, 164, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 226, 228, 230, 232, 234, 236, 238, 240, 243, 245, 247, 249, 251, 253, 254, 256, 258, 260, 262, 264, 266, 268, 270, 273, 275, 277, 278, 280, 282, 283, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 325, 327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351, 352, 353, 354, 355, 356, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 389, 390, 391, 392, 393, 396, 397, 399, 400, 401, 402, 405, 406, 416, 417, 427, 437, 447, 457, 467, 477]
vilc = "[DeviceArray(1.3778734e+08, dtype=float32), DeviceArray(28964586., dtype=float32), DeviceArray(11284699., dtype=float32), DeviceArray(3298409.5, dtype=float32), DeviceArray(3248272.8, dtype=float32), DeviceArray(2751692., dtype=float32), DeviceArray(2742162.2, dtype=float32), DeviceArray(2409707.8, dtype=float32), DeviceArray(2382586., dtype=float32), DeviceArray(2314021., dtype=float32), DeviceArray(2292981.8, dtype=float32), DeviceArray(2287670.8, dtype=float32), DeviceArray(2282602.5, dtype=float32), DeviceArray(2278010.5, dtype=float32), DeviceArray(2232198.2, dtype=float32), DeviceArray(2166271., dtype=float32), DeviceArray(2018294., dtype=float32), DeviceArray(1992428., dtype=float32), DeviceArray(1971308., dtype=float32), DeviceArray(1940991.6, dtype=float32), DeviceArray(1915531.5, dtype=float32), DeviceArray(1810311.1, dtype=float32), DeviceArray(1795056., dtype=float32), DeviceArray(1792706.5, dtype=float32), DeviceArray(1753630.8, dtype=float32), DeviceArray(1743467.8, dtype=float32), DeviceArray(1733797.5, dtype=float32), DeviceArray(1732140.4, dtype=float32), DeviceArray(1665837.6, dtype=float32), DeviceArray(1612797.5, dtype=float32), DeviceArray(1590279.8, dtype=float32), DeviceArray(1582169.9, dtype=float32), DeviceArray(1569345.6, dtype=float32), DeviceArray(1565574.6, dtype=float32), DeviceArray(1564733.2, dtype=float32), DeviceArray(1555978.2, dtype=float32), DeviceArray(1534231., dtype=float32), DeviceArray(1481222.8, dtype=float32), DeviceArray(1454561., dtype=float32), DeviceArray(1418417.5, dtype=float32), DeviceArray(1382321.5, dtype=float32), DeviceArray(1316056.2, dtype=float32), DeviceArray(1237778.2, dtype=float32), DeviceArray(1229314.1, dtype=float32), DeviceArray(1221973.5, dtype=float32), DeviceArray(1165390.6, dtype=float32), DeviceArray(1144010.2, dtype=float32), DeviceArray(1123026., dtype=float32), DeviceArray(1113350.5, dtype=float32), DeviceArray(1108838.2, dtype=float32), DeviceArray(1072999.9, dtype=float32), DeviceArray(1062607.2, dtype=float32), DeviceArray(1051603.4, dtype=float32), DeviceArray(1035472.25, dtype=float32), DeviceArray(1032119.6, dtype=float32), DeviceArray(1005225.4, dtype=float32), DeviceArray(964172.3, dtype=float32), DeviceArray(946888.5, dtype=float32), DeviceArray(918532.1, dtype=float32), DeviceArray(898952., dtype=float32), DeviceArray(876046.5, dtype=float32), DeviceArray(849387.1, dtype=float32), DeviceArray(835321.1, dtype=float32), DeviceArray(833033.4, dtype=float32), DeviceArray(821049.4, dtype=float32), DeviceArray(795349.5, dtype=float32), DeviceArray(788572.1, dtype=float32), DeviceArray(776536.3, dtype=float32), DeviceArray(748571.56, dtype=float32), DeviceArray(735676.44, dtype=float32), DeviceArray(673655.7, dtype=float32), DeviceArray(639063.8, dtype=float32), DeviceArray(599442.25, dtype=float32), DeviceArray(587784., dtype=float32), DeviceArray(583873.9, dtype=float32), DeviceArray(564599.94, dtype=float32), DeviceArray(536698.25, dtype=float32), DeviceArray(530036.5, dtype=float32), DeviceArray(522325.1, dtype=float32), DeviceArray(508257.66, dtype=float32), DeviceArray(492687.25, dtype=float32), DeviceArray(478234.62, dtype=float32), DeviceArray(463144.5, dtype=float32), DeviceArray(457315.94, dtype=float32), DeviceArray(451295., dtype=float32), DeviceArray(444701.62, dtype=float32), DeviceArray(436784.7, dtype=float32), DeviceArray(426904.12, dtype=float32), DeviceArray(417578.97, dtype=float32), DeviceArray(410026.2, dtype=float32), DeviceArray(400450.75, dtype=float32), DeviceArray(388061.75, dtype=float32), DeviceArray(377784.34, dtype=float32), DeviceArray(366507.72, dtype=float32), DeviceArray(354563.75, dtype=float32), DeviceArray(344257.44, dtype=float32), DeviceArray(334452.34, dtype=float32), DeviceArray(327531.75, dtype=float32), DeviceArray(320158.88, dtype=float32), DeviceArray(311358.78, dtype=float32), DeviceArray(301052.5, dtype=float32), DeviceArray(294811.25, dtype=float32), DeviceArray(286658.28, dtype=float32), DeviceArray(276267.44, dtype=float32), DeviceArray(267647.56, dtype=float32), DeviceArray(261725.66, dtype=float32), DeviceArray(255872.6, dtype=float32), DeviceArray(249909.81, dtype=float32), DeviceArray(244101.84, dtype=float32), DeviceArray(238271.1, dtype=float32), DeviceArray(232438.03, dtype=float32), DeviceArray(226828.81, dtype=float32), DeviceArray(221475.02, dtype=float32), DeviceArray(202732.1, dtype=float32), DeviceArray(197709.34, dtype=float32), DeviceArray(192392.73, dtype=float32), DeviceArray(189062.75, dtype=float32), DeviceArray(184998.12, dtype=float32), DeviceArray(179892.75, dtype=float32), DeviceArray(174799.64, dtype=float32), DeviceArray(170919.28, dtype=float32), DeviceArray(168485.38, dtype=float32), DeviceArray(166029.55, dtype=float32), DeviceArray(163590.06, dtype=float32), DeviceArray(161202.69, dtype=float32), DeviceArray(158789.25, dtype=float32), DeviceArray(156107.47, dtype=float32), DeviceArray(152766.88, dtype=float32), DeviceArray(150028.34, dtype=float32), DeviceArray(139773.47, dtype=float32), DeviceArray(126430.5, dtype=float32), DeviceArray(123911.28, dtype=float32), DeviceArray(111499.586, dtype=float32), DeviceArray(101432.59, dtype=float32), DeviceArray(95677.31, dtype=float32), DeviceArray(86758.5, dtype=float32), DeviceArray(83886.94, dtype=float32), DeviceArray(81236.22, dtype=float32), DeviceArray(78824.586, dtype=float32), DeviceArray(74556.21, dtype=float32), DeviceArray(68772.5, dtype=float32), DeviceArray(65955.42, dtype=float32), DeviceArray(62985.066, dtype=float32), DeviceArray(59120.203, dtype=float32), DeviceArray(54344.91, dtype=float32), DeviceArray(50614.93, dtype=float32), DeviceArray(46572.6, dtype=float32), DeviceArray(43170.83, dtype=float32), DeviceArray(40291.06, dtype=float32), DeviceArray(37862.562, dtype=float32), DeviceArray(34672.996, dtype=float32), DeviceArray(32286.355, dtype=float32), DeviceArray(30103.152, dtype=float32), DeviceArray(27959.713, dtype=float32), DeviceArray(25932.717, dtype=float32), DeviceArray(24127.633, dtype=float32), DeviceArray(22540.848, dtype=float32), DeviceArray(21116.668, dtype=float32), DeviceArray(19798.57, dtype=float32), DeviceArray(18548.566, dtype=float32), DeviceArray(17353.334, dtype=float32), DeviceArray(16209.881, dtype=float32), DeviceArray(15117.97, dtype=float32), DeviceArray(14556.338, dtype=float32), DeviceArray(12854.89, dtype=float32), DeviceArray(11416.121, dtype=float32), DeviceArray(10115.647, dtype=float32), DeviceArray(8924.22, dtype=float32), DeviceArray(7410.0605, dtype=float32), DeviceArray(6474.502, dtype=float32), DeviceArray(5903.102, dtype=float32), DeviceArray(5397.9136, dtype=float32), DeviceArray(4852.503, dtype=float32), DeviceArray(4275.8486, dtype=float32), DeviceArray(3740.3145, dtype=float32), DeviceArray(3318.2092, dtype=float32), DeviceArray(3035.0828, dtype=float32), DeviceArray(2840., dtype=float32), DeviceArray(2739.0132, dtype=float32), DeviceArray(2339.2773, dtype=float32), DeviceArray(2134.5425, dtype=float32), DeviceArray(1888.5494, dtype=float32), DeviceArray(1814.9448, dtype=float32), DeviceArray(1411.4485, dtype=float32), DeviceArray(1142.6775, dtype=float32), DeviceArray(1042.7733, dtype=float32), DeviceArray(1004.9304, dtype=float32), DeviceArray(986.5294, dtype=float32), DeviceArray(968.7832, dtype=float32), DeviceArray(952.01074, dtype=float32), DeviceArray(935.4282, dtype=float32), DeviceArray(918.286, dtype=float32), DeviceArray(901.0435, dtype=float32), DeviceArray(883.5505, dtype=float32), DeviceArray(865.9439, dtype=float32), DeviceArray(848.26495, dtype=float32), DeviceArray(830.5609, dtype=float32), DeviceArray(812.79535, dtype=float32), DeviceArray(794.89075, dtype=float32), DeviceArray(776.67413, dtype=float32), DeviceArray(757.87964, dtype=float32), DeviceArray(738.09906, dtype=float32), DeviceArray(716.7239, dtype=float32), DeviceArray(692.82544, dtype=float32), DeviceArray(664.9383, dtype=float32), DeviceArray(630.6442, dtype=float32), DeviceArray(585.80945, dtype=float32), DeviceArray(523.33014, dtype=float32), DeviceArray(432.14005, dtype=float32), DeviceArray(303.7796, dtype=float32), DeviceArray(207.57506, dtype=float32), DeviceArray(137.8675, dtype=float32), DeviceArray(126.48453, dtype=float32), DeviceArray(83.236786, dtype=float32), DeviceArray(71.758286, dtype=float32), DeviceArray(71.57773, dtype=float32), DeviceArray(71.2586, dtype=float32), DeviceArray(71.256874, dtype=float32), DeviceArray(71.249275, dtype=float32), DeviceArray(71.24805, dtype=float32), DeviceArray(71.24785, dtype=float32), DeviceArray(71.24784, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.247795, dtype=float32), DeviceArray(71.24779, dtype=float32), DeviceArray(71.24779, dtype=float32), DeviceArray(71.24779, dtype=float32), DeviceArray(71.24779, dtype=float32), DeviceArray(71.24779, dtype=float32), DeviceArray(71.24779, dtype=float32)]"
vilc = re.split('\(|, ',vilc)
print(len(vilc))
vilc = np.array(vilc)
vilc = vilc[list(range(1, len(vilc), 3))]
print(len(vilc))
vilc = [float(v) for v in vilc]
print(len(vilc))
print(vilc)
plt.semilogy(rilc, vilc)


plt.show()