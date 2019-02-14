
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[0.05967715,0.31272811,0.38120064],
           [0.06924877,0.31399574,0.38927476],
           [0.07760163,0.31529682,0.39730297],
           [0.08514761,0.316598  ,0.405435  ],
           [0.09208529,0.31788774,0.41371669],
           [0.09854105,0.31915964,0.42216771],
           [0.10460016,0.32041147,0.43078451],
           [0.11034519,0.32163392,0.43959946],
           [0.11582861,0.32282438,0.44860838],
           [0.12125195,0.32397403,0.45769997],
           [0.12693587,0.32504188,0.46684403],
           [0.13291976,0.32601609,0.47604771],
           [0.13923109,0.32688978,0.48528853],
           [0.14589733,0.32765488,0.4945453 ],
           [0.15294081,0.32830451,0.50378781],
           [0.16040913,0.3288327 ,0.51294599],
           [0.16829641,0.32923264,0.52200568],
           [0.17663704,0.32950202,0.53087553],
           [0.18542658,0.32963831,0.53950469],
           [0.194683  ,0.32964571,0.54777362],
           [0.20438481,0.32953245,0.55558736],
           [0.21450051,0.32931374,0.56283014],
           [0.225045  ,0.32899214,0.56939832],
           [0.23580516,0.32860685,0.57531639],
           [0.24654454,0.32818541,0.58076148],
           [0.25717098,0.32773171,0.58586819],
           [0.2676619 ,0.32724038,0.59072656],
           [0.27797454,0.32671627,0.59541121],
           [0.28811598,0.32615344,0.59997429],
           [0.2981144 ,0.32554184,0.60444684],
           [0.30796024,0.32488176,0.60886902],
           [0.31765358,0.32417312,0.61326595],
           [0.32725152,0.32339605,0.61764674],
           [0.33672643,0.32255877,0.62203867],
           [0.34612904,0.32164218,0.62644498],
           [0.35543012,0.32065664,0.63088388],
           [0.36469256,0.31957553,0.63535486],
           [0.37389269,0.31840667,0.63987132],
           [0.38304471,0.31714217,0.64443879],
           [0.39217741,0.3157672 ,0.64905581],
           [0.40128101,0.31429179,0.65369346],
           [0.41036026,0.31273137,0.65827048],
           [0.41943089,0.3110759 ,0.66278858],
           [0.42849735,0.30932206,0.66724386],
           [0.43756628,0.30746909,0.67161515],
           [0.44663639,0.30551032,0.67592291],
           [0.45571084,0.30344156,0.68016521],
           [0.46479225,0.30125856,0.68434003],
           [0.4738834 ,0.29895655,0.68844541],
           [0.48298713,0.29653178,0.69247355],
           [0.49210677,0.29397985,0.69641637],
           [0.50124206,0.2912933 ,0.70028445],
           [0.51039442,0.28846644,0.70407555],
           [0.51956604,0.28549239,0.70778757],
           [0.52875737,0.28236507,0.71141814],
           [0.53797032,0.27907651,0.71496511],
           [0.54720633,0.27561835,0.71842627],
           [0.55646618,0.27198194,0.72179933],
           [0.56575179,0.26815831,0.72507686],
           [0.57506396,0.26413616,0.72825856],
           [0.5844019 ,0.25990381,0.73134628],
           [0.59376645,0.25544794,0.7343378 ],
           [0.60315899,0.25075291,0.7372309 ],
           [0.61258023,0.24580185,0.74002332],
           [0.62202867,0.24057822,0.74271284],
           [0.63150634,0.23505861,0.74529729],
           [0.64101385,0.22921789,0.74777447],
           [0.65054975,0.22302965,0.75014233],
           [0.6601155 ,0.21645866,0.75239871],
           [0.66966109,0.20959206,0.75443646],
           [0.67914944,0.2025504 ,0.75607251],
           [0.68856731,0.19536309,0.75727204],
           [0.69789873,0.1880715 ,0.75799944],
           [0.70712705,0.18072781,0.7582183 ],
           [0.71623015,0.17340601,0.75789408],
           [0.72518619,0.16619507,0.75699323],
           [0.73391582,0.15937982,0.7554151 ],
           [0.742438  ,0.15297117,0.75319591],
           [0.7507264 ,0.14712726,0.75031939],
           [0.75875302,0.14202604,0.74677757],
           [0.76641855,0.138124  ,0.74248162],
           [0.77375187,0.13540365,0.73752403],
           [0.78074215,0.13396374,0.73194787],
           [0.78730071,0.13416539,0.72571765],
           [0.79346048,0.13584917,0.71894752],
           [0.79923952,0.13886556,0.71173337],
           [0.8045711 ,0.1433476 ,0.70407306],
           [0.80951556,0.14892862,0.69610138],
           [0.81410623,0.15535762,0.68791286],
           [0.81833226,0.162551  ,0.67954614],
           [0.82221016,0.17035044,0.67105347],
           [0.82580373,0.17848152,0.66252941],
           [0.82913643,0.18683014,0.65401757],
           [0.83223175,0.19530514,0.64555052],
           [0.83511161,0.20383498,0.63715701],
           [0.83779079,0.21239093,0.62882205],
           [0.84047301,0.22054509,0.62062477],
           [0.84315665,0.22835995,0.61252685],
           [0.84584189,0.23587308,0.60452349],
           [0.84852891,0.2431161 ,0.59660988],
           [0.85121789,0.25011607,0.58878075],
           [0.853909  ,0.25689645,0.58103022],
           [0.85660244,0.2634773 ,0.57335309],
           [0.85929842,0.26987598,0.56574457],
           [0.86199717,0.27610765,0.55820021],
           [0.86469879,0.28218652,0.55071307],
           [0.86740342,0.28812467,0.54327796],
           [0.87011129,0.29393237,0.53589157],
           [0.8728224 ,0.29962003,0.52854662],
           [0.8755368 ,0.30519649,0.52123779],
           [0.87825462,0.31066923,0.51396178],
           [0.8809757 ,0.31604633,0.50671071],
           [0.88370002,0.32133421,0.49948046],
           [0.8864275 ,0.32653893,0.49226632],
           [0.88915787,0.33166668,0.48506126],
           [0.89189104,0.33672215,0.47786256],
           [0.89462654,0.34171115,0.47066163],
           [0.89736423,0.34663745,0.46345666],
           [0.9001035 ,0.3515063 ,0.45623879],
           [0.90284414,0.35632081,0.44900682],
           [0.90558537,0.36108585,0.44175118],
           [0.90832688,0.36580418,0.4344705 ],
           [0.91106781,0.37048008,0.42715563],
           [0.91380773,0.37511607,0.41980474],
           [0.91654574,0.37971587,0.41240941],
           [0.91928121,0.38428207,0.40496593],
           [0.92201336,0.38881747,0.39746857],
           [0.92473315,0.39333663,0.38988945],
           [0.92735895,0.39791829,0.38233926],
           [0.92987503,0.40257734,0.37482919],
           [0.93228117,0.40731423,0.36734207],
           [0.93456814,0.4121354 ,0.35989379],
           [0.93672529,0.41704949,0.35249041],
           [0.93874344,0.42206169,0.34515338],
           [0.94061124,0.4271805 ,0.33789036],
           [0.942319  ,0.43241086,0.33072346],
           [0.94385643,0.43775832,0.32367201],
           [0.94520596,0.44323385,0.31677292],
           [0.94635684,0.44884238,0.31004702],
           [0.94730458,0.4545825 ,0.30351343],
           [0.94803956,0.46045697,0.29719868],
           [0.94855339,0.46646676,0.2911334 ],
           [0.94883831,0.47261191,0.28534954],
           [0.94888757,0.47889115,0.27988089],
           [0.94869574,0.4853017 ,0.274763  ],
           [0.94823784,0.4918546 ,0.270085  ],
           [0.94752195,0.49853601,0.26585772],
           [0.94655588,0.50533267,0.26209439],
           [0.94534037,0.51223649,0.2588286 ],
           [0.94387783,0.51923827,0.25609159],
           [0.94216102,0.52633538,0.25394121],
           [0.94018543,0.53352322,0.25242718],
           [0.93797902,0.54077649,0.25151008],
           [0.93554955,0.54808404,0.25120124],
           [0.9329054 ,0.55543497,0.251506  ],
           [0.93005556,0.56281864,0.25242378],
           [0.92698881,0.57023986,0.2539446 ],
           [0.92389095,0.57759334,0.25547121],
           [0.92075602,0.58488757,0.25696852],
           [0.917582  ,0.59212609,0.25843675],
           [0.9143669 ,0.5993122 ,0.25987597],
           [0.91111054,0.6064478 ,0.26128609],
           [0.90782029,0.61352989,0.26266624],
           [0.90448402,0.62056793,0.26401632],
           [0.90109949,0.62756468,0.26533529],
           [0.89766929,0.63451981,0.26662155],
           [0.8941997 ,0.64143088,0.26787324],
           [0.89067607,0.64830784,0.2690882 ],
           [0.88710148,0.65514989,0.27026323],
           [0.88348567,0.66195254,0.27139519],
           [0.87980916,0.6687278 ,0.27247987],
           [0.87608949,0.67546686,0.27351236],
           [0.87231052,0.68217948,0.27448743],
           [0.86848208,0.68886118,0.27539921],
           [0.86459399,0.69551818,0.27624067],
           [0.86065535,0.70214652,0.27700476],
           [0.85664736,0.70875648,0.27768273],
           [0.85259647,0.71533549,0.27826711],
           [0.84847389,0.72189852,0.27874656],
           [0.84429577,0.72843776,0.27912431],
           [0.83991593,0.7349997 ,0.27993458],
           [0.83530235,0.74159177,0.28128714],
           [0.83046407,0.7482071 ,0.28316598],
           [0.82539971,0.75484247,0.28558271],
           [0.82011075,0.76149386,0.28853602],
           [0.81461717,0.76815177,0.29196574],
           [0.80891624,0.77481424,0.29587342],
           [0.803007  ,0.7814787 ,0.30025567],
           [0.7969116 ,0.78813622,0.30503658],
           [0.79061575,0.79478862,0.31024417],
           [0.7841343 ,0.80142979,0.3158173 ],
           [0.77746966,0.80805727,0.32173049],
           [0.77061457,0.8146712 ,0.32798499],
           [0.76358543,0.82126604,0.33450991],
           [0.75636291,0.82784501,0.34133749],
           [0.74896635,0.83440243,0.34838482],
           [0.74137449,0.84094192,0.35568436],
           [0.73360062,0.84745941,0.36317178],
           [0.72563156,0.8539568 ,0.37085045],
           [0.71744707,0.8604384 ,0.37871104],
           [0.70900243,0.86691033,0.38685605],
           [0.70040877,0.87332854,0.39544818],
           [0.69160763,0.87971067,0.40443214],
           [0.68271271,0.88601629,0.4139297 ],
           [0.67367786,0.89225979,0.4238884 ],
           [0.66460539,0.89840845,0.4343887 ],
           [0.65553957,0.90445073,0.44543759],
           [0.64654134,0.91037248,0.45702998],
           [0.63772688,0.91614544,0.46919889],
           [0.62922641,0.92174212,0.48194556],
           [0.62112953,0.92714951,0.49522654],
           [0.61357197,0.93234511,0.50900722],
           [0.6066881 ,0.93730976,0.52323659],
           [0.60060235,0.94202893,0.53784951],
           [0.59546147,0.94648351,0.55279498],
           [0.59131859,0.95067813,0.56796988],
           [0.58822049,0.95461698,0.58328977],
           [0.58618716,0.95830838,0.59867756],
           [0.58519962,0.96176196,0.61414241],
           [0.58589491,0.96485048,0.62968128],
           [0.58858572,0.96752083,0.64515418],
           [0.59346778,0.96974118,0.66039099],
           [0.60057056,0.9715099 ,0.67519087],
           [0.60972899,0.97286003,0.68936393],
           [0.62061614,0.9738533 ,0.70277101],
           [0.63282011,0.97456677,0.71534456],
           [0.6460197 ,0.97505802,0.72707071],
           [0.65983295,0.97539946,0.73800031],
           [0.67391363,0.9756605 ,0.74821436],
           [0.68824425,0.97583931,0.75775384],
           [0.70249287,0.97600939,0.76672981],
           [0.71672724,0.97615333,0.77517435],
           [0.73078248,0.97631087,0.78316582],
           [0.74461339,0.97649479,0.7907562 ],
           [0.75821414,0.9767085 ,0.79798694],
           [0.77162405,0.97694387,0.80488053],
           [0.78477371,0.97722182,0.81148327],
           [0.79766702,0.97754437,0.81782117],
           [0.81030325,0.97791521,0.82391463],
           [0.82270569,0.97833008,0.82978729],
           [0.83500389,0.97874291,0.83554496],
           [0.84703236,0.97920375,0.84125493],
           [0.85879878,0.97971517,0.8469129 ],
           [0.87032091,0.98027591,0.85251705],
           [0.88161897,0.98088337,0.8580657 ],
           [0.89271186,0.98153506,0.86355599],
           [0.90356037,0.98225004,0.86896795],
           [0.91424589,0.98300213,0.87431377],
           [0.92477919,0.983791  ,0.87958411],
           [0.93513416,0.98463071,0.8847581 ],
           [0.94537207,0.98550138,0.88983578],
           [0.95552072,0.98639563,0.89480545],
           [0.96559862,0.9873095 ,0.89965153],
           [0.975673  ,0.98821889,0.90436862],
           [0.98579195,0.98910616,0.90894944],
           [0.99631677,0.98982363,0.9134405 ]]
test_cm = ListedColormap(cm_data, name="amstelveen")