
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[0.97776415,0.98285354,0.99303202],
           [0.97571151,0.97828261,0.98479303],
           [0.97352334,0.97375272,0.97681425],
           [0.97118693,0.96927404,0.96905213],
           [0.96864451,0.96487555,0.9614532 ],
           [0.96581266,0.96060763,0.95385579],
           [0.96275297,0.95649019,0.94583601],
           [0.95982718,0.95242369,0.93696276],
           [0.95735433,0.94827496,0.92728016],
           [0.95534027,0.9440089 ,0.91708239],
           [0.95366858,0.93964868,0.90657342],
           [0.9522413 ,0.93522164,0.89584983],
           [0.9509899 ,0.93074774,0.88497038],
           [0.94987536,0.92623909,0.87395748],
           [0.94886894,0.92170393,0.86283316],
           [0.94798351,0.91714169,0.85153709],
           [0.94718443,0.91255946,0.84012899],
           [0.94645113,0.90796168,0.82864043],
           [0.94577944,0.90334941,0.81706886],
           [0.94520757,0.89871478,0.8053172 ],
           [0.94467729,0.89406923,0.79351119],
           [0.94419103,0.88941209,0.7816386 ],
           [0.94379339,0.88473334,0.76959134],
           [0.94341887,0.88004657,0.75751441],
           [0.94310579,0.87534287,0.74531503],
           [0.94284528,0.87062354,0.73300891],
           [0.94259936,0.86589683,0.72067965],
           [0.94244028,0.86114508,0.70815553],
           [0.94228647,0.85638077,0.69569626],
           [0.94224699,0.85155253,0.6833315 ],
           [0.94234493,0.84666555,0.67085807],
           [0.94258413,0.84171227,0.65832152],
           [0.94297459,0.83667705,0.64583326],
           [0.94352398,0.83156881,0.63321135],
           [0.94422948,0.82637057,0.62065346],
           [0.94510351,0.82107653,0.60813001],
           [0.9461595 ,0.81568446,0.59555158],
           [0.9473919 ,0.81018527,0.58303016],
           [0.94881665,0.80455832,0.57069775],
           [0.9504325 ,0.79880843,0.55845362],
           [0.95224921,0.79292316,0.54635122],
           [0.95426204,0.786895  ,0.53447274],
           [0.95646808,0.78071549,0.52290183],
           [0.95885018,0.77437946,0.51177984],
           [0.96138505,0.76788914,0.50119554],
           [0.96404118,0.76125254,0.49122398],
           [0.96676984,0.75448455,0.48198579],
           [0.96951066,0.74760849,0.47359136],
           [0.97219883,0.74065356,0.46612634],
           [0.97476982,0.7336529 ,0.45964299],
           [0.97716262,0.72664205,0.45415578],
           [0.97933799,0.7196485 ,0.44963691],
           [0.98126175,0.71269967,0.44603047],
           [0.98292153,0.70581254,0.44325646],
           [0.98431289,0.69900032,0.44122307],
           [0.98544524,0.69226803,0.43983547],
           [0.98632871,0.68561988,0.43899947],
           [0.98697272,0.67905742,0.43866171],
           [0.98738932,0.67257107,0.4388983 ],
           [0.98760433,0.66616478,0.43946216],
           [0.98763215,0.65983723,0.44028085],
           [0.98748405,0.65357637,0.44145229],
           [0.98717118,0.64738093,0.44292056],
           [0.98671795,0.64125101,0.44449232],
           [0.98612346,0.63517546,0.44634451],
           [0.98540542,0.62915335,0.44835   ],
           [0.98458225,0.62318226,0.45039487],
           [0.98364324,0.61725039,0.45273469],
           [0.98261869,0.61136646,0.4550084 ],
           [0.98149896,0.60551308,0.457523  ],
           [0.98031483,0.59969856,0.45992104],
           [0.97907943,0.59390543,0.46230578],
           [0.97782536,0.58810792,0.46469926],
           [0.97655098,0.58230671,0.4671022 ],
           [0.97522758,0.5765239 ,0.46950639],
           [0.97392188,0.57070558,0.47194437],
           [0.97252512,0.56493903,0.47436836],
           [0.97113943,0.55914074,0.47682921],
           [0.96971786,0.55334804,0.47931072],
           [0.96824534,0.54757337,0.48180759],
           [0.96677293,0.54177262,0.48434903],
           [0.96523821,0.53599743,0.48691788],
           [0.96366261,0.53022974,0.48952153],
           [0.96207176,0.52444558,0.49218178],
           [0.96042181,0.5186821 ,0.49488937],
           [0.95870277,0.51294835,0.49764223],
           [0.95694894,0.50721162,0.50047167],
           [0.95514737,0.50148273,0.50337684],
           [0.95326332,0.49579248,0.50635535],
           [0.95128254,0.49015347,0.50941541],
           [0.94922179,0.48454934,0.5125761 ],
           [0.94706641,0.47899225,0.51585174],
           [0.94479598,0.47350158,0.51924557],
           [0.94239074,0.46809518,0.52277399],
           [0.93982462,0.46279912,0.52644075],
           [0.93707174,0.45763743,0.53027032],
           [0.93409769,0.45264591,0.53426565],
           [0.93086519,0.44786239,0.53844199],
           [0.92733044,0.44333265,0.5428085 ],
           [0.92342268,0.43913238,0.54735577],
           [0.91903019,0.43538111,0.5520578 ],
           [0.91414968,0.43208875,0.55690097],
           [0.90863999,0.42941335,0.56178445],
           [0.90247932,0.42739061,0.56661179],
           [0.89559886,0.42610931,0.57121395],
           [0.88808962,0.42548912,0.57545905],
           [0.88028306,0.42518848,0.57938213],
           [0.8723156 ,0.42505563,0.58306464],
           [0.86416964,0.42511026,0.58645798],
           [0.85586012,0.4253342 ,0.5895531 ],
           [0.84740205,0.42570965,0.59234038],
           [0.8387818 ,0.42625173,0.59476999],
           [0.83003731,0.42691708,0.59686239],
           [0.82117454,0.42769895,0.5985964 ],
           [0.81220173,0.42858799,0.59995569],
           [0.80315371,0.4295459 ,0.60096469],
           [0.79402586,0.43057745,0.60159687],
           [0.78485256,0.43164561,0.60188418],
           [0.77564233,0.43274155,0.60182487],
           [0.76641058,0.43384873,0.60143439],
           [0.75717318,0.43495072,0.60073036],
           [0.74794065,0.43603683,0.59972625],
           [0.73872305,0.43709693,0.59843746],
           [0.7295399 ,0.4381124 ,0.59689324],
           [0.72037962,0.43909388,0.59508581],
           [0.71127023,0.44001519,0.59305994],
           [0.70221469,0.44087312,0.59083467],
           [0.6931934 ,0.44165048,0.58862883],
           [0.68424065,0.44232522,0.58644691],
           [0.6753542 ,0.44290302,0.58428029],
           [0.66653014,0.44339011,0.58212205],
           [0.65776586,0.44379142,0.5799662 ],
           [0.64906813,0.44410588,0.57780611],
           [0.64042845,0.44434149,0.57563865],
           [0.63185913,0.4444936 ,0.57345787],
           [0.62335422,0.44456829,0.57126166],
           [0.61492116,0.44456372,0.5690463 ],
           [0.60655996,0.44448224,0.56680997],
           [0.59827766,0.44432222,0.56455031],
           [0.59007489,0.44408556,0.56226667],
           [0.5819568 ,0.44377163,0.5599583 ],
           [0.57390781,0.44339019,0.55763107],
           [0.56593417,0.44293967,0.55528454],
           [0.55805654,0.44241146,0.55291567],
           [0.55027358,0.44180807,0.55052741],
           [0.54258091,0.44113315,0.5481251 ],
           [0.53496203,0.44039558,0.54571849],
           [0.52744256,0.4395847 ,0.54330441],
           [0.52003107,0.43869785,0.54088718],
           [0.51269173,0.43775221,0.5384825 ],
           [0.50544428,0.43673962,0.5360921 ],
           [0.49829788,0.43565708,0.53371962],
           [0.49122265,0.43451802,0.53138094],
           [0.48422371,0.43332056,0.52908205],
           [0.47731158,0.43206105,0.52682502],
           [0.47044619,0.43075599,0.52462916],
           [0.46363842,0.42940118,0.52249571],
           [0.45689108,0.42799609,0.52042629],
           [0.45014761,0.42656212,0.51844596],
           [0.44344276,0.42508654,0.51654082],
           [0.43674074,0.42358261,0.51472509],
           [0.43002453,0.42205631,0.51300425],
           [0.42330677,0.42050346,0.51136795],
           [0.41652591,0.41894456,0.50984202],
           [0.40971783,0.41736764,0.50840135],
           [0.40283181,0.41578867,0.50706549],
           [0.39587715,0.41420416,0.50582149],
           [0.38880103,0.41262977,0.50468618],
           [0.38165094,0.41105722,0.50357076],
           [0.37433389,0.40950978,0.50253687],
           [0.36695129,0.40796273,0.50147526],
           [0.3593755 ,0.40644509,0.5004852 ],
           [0.35165808,0.40494413,0.49949889],
           [0.3437852 ,0.40346176,0.49851084],
           [0.33568057,0.40201315,0.4975628 ],
           [0.32740888,0.40058268,0.49657743],
           [0.31893143,0.3991765 ,0.49556525],
           [0.31018543,0.39780514,0.49454707],
           [0.30115597,0.39646858,0.49350214],
           [0.291929  ,0.39514732,0.49233917],
           [0.2823928 ,0.39386008,0.49110169],
           [0.27252905,0.39260629,0.48976325],
           [0.26232247,0.39138482,0.4882881 ],
           [0.25176744,0.39019236,0.48663269],
           [0.24086354,0.38902483,0.48474636],
           [0.229611  ,0.38787869,0.48257038],
           [0.21803724,0.3867461 ,0.48003396],
           [0.20620953,0.38561391,0.47705515],
           [0.19438358,0.3844467 ,0.47348055],
           [0.18286662,0.383206  ,0.46919834],
           [0.17195197,0.38185727,0.46418109],
           [0.16255027,0.38029948,0.45826577],
           [0.15479279,0.37852212,0.45166983],
           [0.14933848,0.37645942,0.44442085],
           [0.14574011,0.37416134,0.43684391],
           [0.14406538,0.37162336,0.42899911],
           [0.14378511,0.36889985,0.42106373],
           [0.14452195,0.36602865,0.41313149],
           [0.14601618,0.36303626,0.40524046],
           [0.14806564,0.35994458,0.39740011],
           [0.14998752,0.35684736,0.38963735],
           [0.15149384,0.35378642,0.38198437],
           [0.15259799,0.35076395,0.37443712],
           [0.15331414,0.34778201,0.3669862 ],
           [0.15364863,0.34484288,0.35962655],
           [0.15360118,0.34194954,0.35235222],
           [0.15316109,0.33910603,0.34515887],
           [0.15234502,0.33631248,0.33803253],
           [0.15115175,0.33357103,0.33096235],
           [0.14957679,0.33088366,0.32393916],
           [0.14761386,0.32825217,0.31695282],
           [0.14522869,0.32568153,0.31000097],
           [0.14240971,0.32317309,0.30307282],
           [0.13916336,0.32072495,0.29615256],
           [0.13546502,0.31833877,0.28922995],
           [0.13123196,0.3160218 ,0.28231858],
           [0.12648908,0.31376689,0.27539175],
           [0.12115928,0.31157788,0.26845528],
           [0.11516424,0.30945633,0.26151259],
           [0.10845886,0.30739743,0.25455333],
           [0.10085859,0.30540783,0.24760257],
           [0.09227286,0.30347897,0.24064583],
           [0.08237615,0.30161751,0.23371569],
           [0.07084867,0.299816  ,0.22680662],
           [0.05693174,0.29808103,0.21992898],
           [0.04109745,0.29630508,0.21288276],
           [0.02544515,0.29440217,0.20556594],
           [0.01334206,0.29230069,0.19795456],
           [0.00654296,0.28987788,0.19007725],
           [0.00764801,0.28693149,0.18222915],
           [0.01764745,0.28332901,0.17515753],
           [0.03314575,0.27924533,0.16937901],
           [0.0498437 ,0.27491283,0.1647552 ],
           [0.06362224,0.27049384,0.16090287],
           [0.07506084,0.26604363,0.15758019],
           [0.0847532 ,0.26158827,0.15463146],
           [0.09314782,0.25713284,0.15196877],
           [0.10043475,0.25269647,0.14950173],
           [0.10683313,0.2482799 ,0.14718742],
           [0.11254625,0.24387364,0.14501189],
           [0.11762734,0.23948883,0.14292801],
           [0.12232261,0.2350957 ,0.14088943],
           [0.12655887,0.23072267,0.13882103],
           [0.13038959,0.22636883,0.136721  ],
           [0.13381611,0.22204335,0.13458383],
           [0.13693264,0.21773242,0.13241182],
           [0.13975729,0.21343794,0.13020126],
           [0.14230063,0.20916318,0.12794802],
           [0.14460499,0.20490221,0.12565033],
           [0.14669916,0.20065138,0.12330513],
           [0.14859013,0.19641244,0.12090796],
           [0.15030148,0.19218177,0.11845546],
           [0.15187751,0.18794833,0.11594475],
           [0.15332328,0.1837121 ,0.11337144],
           [0.15485185,0.17940288,0.11073155]]
test_cm = ListedColormap(cm_data, name="coldmorning")