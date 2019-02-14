
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[0.99750501,0.98216417,0.98506961],
           [0.99516601,0.97735606,0.97900677],
           [0.99280331,0.97254084,0.97326894],
           [0.99098363,0.96752078,0.96755316],
           [0.98986404,0.96228956,0.96128019],
           [0.9890732 ,0.95698959,0.95449626],
           [0.98842498,0.95167176,0.94742219],
           [0.98787452,0.94634195,0.94016651],
           [0.98740373,0.94100103,0.93278284],
           [0.98699732,0.93565098,0.92530485],
           [0.9866575 ,0.93028907,0.91774172],
           [0.98636893,0.92491885,0.91011377],
           [0.98613654,0.91953752,0.90242042],
           [0.98595145,0.91414701,0.89467213],
           [0.98581135,0.90874721,0.88687248],
           [0.98571165,0.90333885,0.87902657],
           [0.98565046,0.89792183,0.87113659],
           [0.98562466,0.89249652,0.86320563],
           [0.98563637,0.88706147,0.85523151],
           [0.98559423,0.88166273,0.84716072],
           [0.98551106,0.87630302,0.83890996],
           [0.98536631,0.8709928 ,0.83046839],
           [0.98513587,0.8657443 ,0.82182667],
           [0.98479202,0.86057147,0.81297758],
           [0.9843035 ,0.85548997,0.80391683],
           [0.98362488,0.85052177,0.79464758],
           [0.98271062,0.84568842,0.78518224],
           [0.98152591,0.84100586,0.77554259],
           [0.98000415,0.83650389,0.76576712],
           [0.9781156 ,0.83219393,0.75590519],
           [0.97581486,0.82809391,0.74601754],
           [0.97309122,0.82420471,0.73616732],
           [0.96993444,0.82052687,0.72641633],
           [0.96636001,0.81704921,0.71681472],
           [0.96240149,0.81375254,0.70739983],
           [0.95807012,0.8106291 ,0.69819928],
           [0.95341729,0.80765284,0.68922376],
           [0.94847702,0.80480654,0.68047762],
           [0.94328129,0.80207441,0.67195888],
           [0.9378605 ,0.79944169,0.66366139],
           [0.93224235,0.7968953 ,0.65557478],
           [0.92645295,0.79442283,0.64768944],
           [0.92051026,0.79201596,0.63999408],
           [0.91442434,0.78967008,0.63247823],
           [0.90822683,0.7873706 ,0.62512864],
           [0.90192517,0.7851141 ,0.6179367 ],
           [0.89552468,0.78289826,0.61089236],
           [0.88904983,0.78071204,0.60398355],
           [0.88249244,0.77855916,0.59720407],
           [0.87587583,0.77642913,0.59054345],
           [0.86919863,0.7743225 ,0.58399617],
           [0.86245031,0.77224   ,0.57760682],
           [0.85562622,0.77018487,0.57135329],
           [0.84871561,0.76815826,0.5652741 ],
           [0.8417272 ,0.76615743,0.55934484],
           [0.83465438,0.76418281,0.5535889 ],
           [0.82749966,0.76223266,0.54800475],
           [0.8202629 ,0.76030582,0.54259806],
           [0.81294281,0.75840129,0.53737929],
           [0.80554252,0.75651715,0.53234526],
           [0.79806027,0.75465232,0.52750951],
           [0.79049995,0.75280462,0.52286803],
           [0.7828618 ,0.75097247,0.51842941],
           [0.77514838,0.74915385,0.51419535],
           [0.76736367,0.74734648,0.51016311],
           [0.75950743,0.74554872,0.506345  ],
           [0.75158847,0.74375752,0.50272255],
           [0.74360277,0.74197185,0.49932153],
           [0.73556416,0.74018798,0.49610574],
           [0.72746462,0.73840551,0.49311508],
           [0.71932061,0.73662053,0.49030296],
           [0.71112621,0.73483241,0.4877014 ],
           [0.70289417,0.73303808,0.48527478],
           [0.69462161,0.73123661,0.48304288],
           [0.68631869,0.7294256 ,0.48097705],
           [0.67798366,0.72760418,0.47909068],
           [0.66962588,0.72577032,0.47735585],
           [0.66124278,0.72392344,0.47578719],
           [0.65284425,0.72206191,0.47435002],
           [0.64442405,0.72018573,0.47306934],
           [0.63599349,0.7182935 ,0.47190189],
           [0.62754733,0.71638499,0.47087118],
           [0.61909184,0.71445974,0.46994675],
           [0.61062613,0.71251778,0.46912653],
           [0.60214948,0.71055866,0.46841501],
           [0.59366546,0.70858271,0.46778415],
           [0.58517115,0.70658986,0.46724319],
           [0.57666474,0.70458075,0.46678462],
           [0.56815009,0.70255511,0.46638875],
           [0.55962184,0.70051386,0.46605962],
           [0.55107671,0.69845734,0.46579978],
           [0.54251669,0.6963859 ,0.46558811],
           [0.53393782,0.69430039,0.46542133],
           [0.52533495,0.69220143,0.46530375],
           [0.51670439,0.69008957,0.46523313],
           [0.50805878,0.6879597 ,0.46522542],
           [0.4994232 ,0.68580109,0.46533346],
           [0.49080485,0.68361146,0.46556114],
           [0.48224601,0.68137853,0.46592901],
           [0.47372666,0.67910892,0.46642027],
           [0.46526168,0.67679928,0.46703304],
           [0.45689297,0.6744389 ,0.46778144],
           [0.44862788,0.67202748,0.46865696],
           [0.44046879,0.6695664 ,0.46964573],
           [0.43247378,0.66704231,0.4707617 ],
           [0.42463931,0.66445903,0.47198485],
           [0.4169777 ,0.66181633,0.47330139],
           [0.40955742,0.65909999,0.47472391],
           [0.40234603,0.65632183,0.47621585],
           [0.39539246,0.65347326,0.47777891],
           [0.38871295,0.65055409,0.4793994 ],
           [0.38229088,0.64757182,0.48105215],
           [0.37619113,0.64451475,0.48274493],
           [0.37037684,0.64139477,0.48444813],
           [0.36484627,0.63821529,0.48614658],
           [0.35965778,0.63496587,0.4878515 ],
           [0.35475479,0.63166189,0.48953013],
           [0.3501344 ,0.62830586,0.49117636],
           [0.34582412,0.6248937 ,0.49279263],
           [0.34180006,0.62143224,0.49436589],
           [0.33803763,0.6179279 ,0.49588662],
           [0.33452766,0.61438359,0.49735033],
           [0.33126488,0.61080113,0.49875531],
           [0.32825829,0.60717918,0.50010448],
           [0.32546719,0.60352662,0.50138688],
           [0.32287936,0.59984609,0.50260167],
           [0.32048216,0.59614026,0.50374746],
           [0.318263  ,0.59241157,0.50482432],
           [0.31620926,0.58866241,0.50583228],
           [0.31430832,0.5848951 ,0.50677135],
           [0.31254705,0.58111215,0.50764013],
           [0.31091421,0.57731527,0.50844207],
           [0.30939752,0.57350663,0.50917694],
           [0.3079851 ,0.56968828,0.5098452 ],
           [0.30666674,0.5658617 ,0.51045   ],
           [0.30543024,0.56202916,0.51098993],
           [0.30426656,0.55819185,0.51146893],
           [0.30316537,0.55435151,0.51188771],
           [0.30211716,0.55050967,0.51224784],
           [0.30111395,0.54666742,0.5125526 ],
           [0.30014621,0.54282644,0.51280224],
           [0.29920648,0.53898781,0.51299949],
           [0.29830015,0.53514987,0.51314976],
           [0.29742072,0.53131357,0.51325518],
           [0.29654969,0.5274825 ,0.51331513],
           [0.29568066,0.52365766,0.5133316 ],
           [0.29480873,0.51983966,0.51330803],
           [0.29392865,0.51602929,0.51324683],
           [0.29306169,0.51222148,0.51315558],
           [0.29218356,0.50842118,0.51303298],
           [0.29128488,0.50463011,0.51288063],
           [0.29036216,0.50084872,0.51270125],
           [0.28943875,0.49706994,0.51251672],
           [0.28852138,0.49329582,0.51229328],
           [0.28760157,0.48952612,0.51204924],
           [0.28668162,0.48576021,0.51178537],
           [0.28576391,0.48199744,0.51150219],
           [0.28486977,0.47823501,0.51118301],
           [0.28398388,0.4744742 ,0.51084463],
           [0.28310772,0.47071441,0.51048855],
           [0.28224454,0.46695475,0.5101152 ],
           [0.2813979 ,0.46319423,0.50972506],
           [0.28057762,0.45943142,0.50931018],
           [0.27978723,0.45566546,0.50886961],
           [0.27902138,0.45189644,0.50841084],
           [0.27828396,0.44812325,0.5079339 ],
           [0.27757831,0.44434498,0.50743796],
           [0.27690702,0.44056097,0.50692122],
           [0.27627344,0.43677029,0.50638262],
           [0.27568153,0.43297174,0.50582161],
           [0.27513452,0.42916474,0.50523375],
           [0.2746403 ,0.42534773,0.50461354],
           [0.27419443,0.42152073,0.50396559],
           [0.2737992 ,0.41768302,0.50328766],
           [0.27345747,0.41383363,0.50257804],
           [0.27317005,0.40997228,0.50183326],
           [0.27293949,0.406098  ,0.50105155],
           [0.27276623,0.40221052,0.50022942],
           [0.27265137,0.3983093 ,0.49936418],
           [0.2725957 ,0.39439383,0.49845305],
           [0.27259837,0.39046411,0.49749227],
           [0.27266009,0.38651955,0.49647945],
           [0.27277907,0.38256039,0.49541072],
           [0.27295397,0.37858664,0.49428286],
           [0.27318397,0.37459807,0.49309325],
           [0.273466  ,0.37059521,0.49183823],
           [0.27379753,0.36657835,0.49051484],
           [0.27417571,0.36254785,0.48912026],
           [0.27459737,0.35850416,0.48765187],
           [0.27505847,0.354448  ,0.486107  ],
           [0.27555906,0.35037984,0.48447661],
           [0.27609095,0.34630065,0.48276531],
           [0.27664964,0.34221125,0.48097152],
           [0.27723052,0.33811252,0.47909395],
           [0.27782903,0.33400533,0.4771317 ],
           [0.27844058,0.32989055,0.47508419],
           [0.27906033,0.32576918,0.4729511 ],
           [0.27968407,0.321642  ,0.47073262],
           [0.2803075 ,0.31750984,0.46842922],
           [0.28092641,0.31337352,0.46604164],
           [0.28153651,0.30923392,0.46357092],
           [0.28213464,0.30509151,0.46101852],
           [0.28271762,0.3009468 ,0.4583861 ],
           [0.2832811 ,0.29680076,0.45567538],
           [0.28382339,0.29265337,0.45288855],
           [0.28434196,0.28850493,0.45002784],
           [0.28483346,0.28435612,0.44709558],
           [0.28529744,0.28020643,0.4440943 ],
           [0.28573059,0.27605654,0.44102655],
           [0.28613282,0.27190584,0.43789493],
           [0.28650206,0.26775451,0.43470211],
           [0.28683807,0.26360198,0.43145068],
           [0.28713915,0.2594483 ,0.42814332],
           [0.28740626,0.25529237,0.42478242],
           [0.2876365 ,0.25113474,0.42137081],
           [0.2878319 ,0.24697384,0.41791059],
           [0.28799151,0.24280933,0.4144042 ],
           [0.28811475,0.23864074,0.41085396],
           [0.28820584,0.23446755,0.40724934],
           [0.28827376,0.23028269,0.40360109],
           [0.288322  ,0.22607159,0.39997233],
           [0.2883482 ,0.22183203,0.39637393],
           [0.28835384,0.2175621 ,0.39280405],
           [0.28834138,0.21325921,0.38926062],
           [0.28830892,0.20892285,0.38574233],
           [0.28826033,0.20454937,0.38224671],
           [0.28819295,0.2001384 ,0.37877246],
           [0.28811004,0.19568616,0.37531716],
           [0.28800931,0.19119177,0.37187932],
           [0.28789206,0.18665209,0.36845682],
           [0.28775903,0.18206403,0.36504766],
           [0.28760888,0.17742546,0.36165011],
           [0.28744172,0.17273309,0.35826226],
           [0.2872578 ,0.16798316,0.35488217],
           [0.28705709,0.16317164,0.35150797],
           [0.28683914,0.15829433,0.34813788],
           [0.28660028,0.15334888,0.34477055],
           [0.28633731,0.14833229,0.34140459],
           [0.28605552,0.14323436,0.33803757],
           [0.28575524,0.13804752,0.33466771],
           [0.28542772,0.13277088,0.33129447],
           [0.2850752 ,0.12739354,0.32791592],
           [0.28470255,0.12190046,0.3245298 ],
           [0.28429448,0.11629336,0.32113662],
           [0.28386666,0.11054303,0.31773272],
           [0.2834035 ,0.10464759,0.31431877],
           [0.28291042,0.09858133,0.31089259],
           [0.28238793,0.09231844,0.30745265],
           [0.28183145,0.08583308,0.30399827],
           [0.2812379 ,0.07909032,0.30052857],
           [0.28060929,0.07203746,0.29704193],
           [0.27995087,0.06459911,0.29353606],
           [0.27924742,0.05670541,0.29001228],
           [0.27851139,0.0482031 ,0.28646705],
           [0.27774477,0.03887069,0.2828985 ],
           [0.27703799,0.02905358,0.27928645]]
test_cm = ListedColormap(cm_data, name="nearbow")