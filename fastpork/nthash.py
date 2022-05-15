# ToDos:
# k fix? generator function
# alle 4er fix?
import numpy as np
from numba import njit, uint8, uint64
from math import ceil

tetramerTab = np.array([
6047278271377325800, 6842100033257738704, 5716751207778949560, 5058261232784932554,
5322212292231585944, 4955210659836481440, 6153481158060361672, 6630136099103187130,
7683058811908681801, 7460089081761259377, 8513615477720831769, 9169618076073996395,
8669810821731892908, 8451393064794886548, 7271235746105367036, 7894785163577458318,
7461575445318369801, 7680024275870068017, 8878022265940976985, 8237757801848291883,
9060296013225843833, 8116780716040188737, 6991106539262573353, 7521593563379047515,
6845292839028968616, 6045914992845185936, 4775672622745250808, 5413871935584767114,
5490367161684853325, 4695435745326017909, 5803018666222232861, 6480400171096490607,
2381043025085637546, 3175899973157948562, 4445879008075678970, 3807116472585741192,
4268108881087626714, 3901072061426881250, 2847008385469766282, 3379366782720458232,
1763336001516006667, 1540401457157816883, 342666797974407771, 983493939256405289,
771890739233563630, 553508169276984534, 1589643033626739902, 2263336780810576844,
330722743541775969, 688712796851212633, 1742668713148160305, 1245320973785726531,
2208596672445898769, 1422777727841816361, 152919646732699457, 826464124477841459,
4460107693596700864, 3530055095011467256, 2403999925630162832, 2899137386794791138,
3398970977768160805, 2464498338584432925, 3716128830812494197, 4248337413163712007,
4264326372183459627, 3906261395711551507, 2851952150714671227, 3383149429014333193,
2386233046276708699, 3172117876357805667, 4441779805226941963, 3801926588820052345,
170684860043692426, 1100671402695403186, 2226926226858061530, 1693589575942097320,
1193606390847620975, 2128144916583147607, 876319371625685055, 382305650241144653,
1102545060664966090, 168107437338776818, 1437989166537956506, 1915072878734195688,
1548519783094789562, 1757891215679916674, 703889661060612842, 46092416782165400,
3908715595921208683, 4262294307145226835, 3064498623987880507, 2585134797421409609,
2661735585529691022, 3019760716990469302, 4055956603131813086, 3543998858204232620,
5317339067591416425, 4959238909506745681, 6157334207435046201, 6635009461133220427,
6051307208490845209, 6837227221258447649, 5711490920986878793, 5054232433096901691,
8122648135453742280, 9052599496358476784, 7782418148093113240, 7307023562816214250,
7095314801322056237, 8029818144085865749, 9137340041034366333, 8622472983995947535,
7806751516869674914, 7011855109925922970, 8137690373747335410, 8757695200062998400,
8531879593853721042, 8898947385530005226, 7700757522090507906, 7186022138009770480,
6135219772853324035, 6358123720871388731, 5304510851123850835, 4682089562405882145,
5182028715320330214, 5400512630465816798, 6580751683450298550, 5923625422568720324,
13124074928584983660, 13491146941631638356, 12293650504952193852, 11816502978180760654,
12399079312662682140, 11604187204414436644, 12730450818222161228, 13388307479092468286,
10327209524901530317, 9388215691182564853, 10657868830410829213, 11137168911054473967,
11357920004770333736, 10414374197647485712, 9306325182584103800, 9818342344138146826,
9386341947321596045, 10329786896059045813, 11455812913355464669, 10924692575052363951,
10984992149858150141, 10766613702172592581, 9568826821541020077, 10208598699842184927,
13488692655530571308, 13126106942075820308, 12072096584926548348, 12605510244625659406,
12249677498819492041, 11882645355480553457, 13062230760632229785, 13556163143878539499,
14178740190036597038, 14545847390080448022, 15599559227675164286, 15067834145139579148,
16065876409530435422, 15270949115358734438, 14000758968863088654, 14640014089599289212,
18281953465151117199, 17342994818563569847, 16217267316526477535, 16746698532205467565,
17255653680509032810, 16312143059561297490, 17564497017566543418, 18061360711745100104,
16237972021990524133, 17023861349393640413, 18293930539975648181, 17619893477009409223,
18115916316835994261, 17757855915011241389, 16704251839199542725, 17200966263939144375,
15576639675766950468, 15362743113290245500, 14164544455910714644, 14841019967217601126,
14620295210399335585, 14410818688327658393, 15446357621659116529, 16085462927495578755,
18237799192036655099, 17294270664133710019, 16258109964509321387, 16773410497518403545,
16657084189963477387, 16875519862962278067, 18127020052323321563, 17507580374969491881,
14153168177888129370, 14515696771658964578, 15624080140268688906, 15110866744451150200,
15466708232756051903, 15833797605570023559, 14563810316809509103, 14085706539145691037,
14517711175708869402, 14150731501263563810, 15402451490950456394, 15899948742203982648,
15224753927964908906, 16019597712369578578, 14983744703118572090, 14310050713553640776,
17296865610423782843, 18235907873078829699, 17055988043521714923, 16561000163437350297,
16340222631939670878, 17283720110790814822, 18338064546595415054, 17805706452459078524,
10375933128878629561, 9432369415202180481, 10612588863825479145, 11105888166746317467,
10794790039591648457, 11013260899437695985, 9905396050428550041, 9228014311730625771,
13154226096333843480, 13516719503928509216, 12264699899470662472, 11768891770841246778,
11836546934201131773, 12203601119882644933, 13328994472388527533, 12798507759874630367,
12277767672444305266, 12068343612890878026, 13176021535246260258, 13816435502572994384,
12705517425460601090, 13640043170446921274, 12460006250421962322, 11929369723008524576,
10597232027372843475, 11387585128312430315, 10351852510211364483, 9713802769929286129,
9357917249443839798, 10143859113470169102, 11342251114164164710, 10664720106027613972
], dtype=np.uint64)

trimerTab = np.array([
13237172352163388750, 13451082378889146998, 12324706752351386142, 11704099346423635308,
12503002411303846718, 11573033083854154758, 12770611021816489070, 13284814289517544220,
10286336837755622383, 9500434588327378135, 10554658215321236671, 11177611689138066381,
11245073286936829194, 10454751004568891954, 9274956656780491354, 9930495270120774952,
9498947889754972591, 10289371588586147479, 11487222103436658431, 10812501148518244749,
11088845979783725023, 10735249574334615783, 9609199230360475791, 10105458452942995453,
13447889238169808654, 13238535845420384310, 11968673763542288478, 12645600078955589420,
12136759312206930411, 11922809957208297171, 13031072242070652603, 13668666814620918217,
14219262150204358668, 14433136993975185204, 15703263506252408668, 15026899868095529006,
16097136083696541308, 15167201938128040260, 14113514427211577644, 14608043031429815902,
18169629015343943341, 17383691583363408277, 16185576633819064829, 16859734366019948175,
17215452794964541512, 16425095330967072624, 17460550829194815256, 18101973914136232042,
16197524846324948423, 17136496960994620159, 18190301010467109527, 17660752969549176293,
18084590689685816247, 17861669045228104847, 16591430392433501415, 17233003275094786965,
15689030113991676774, 15321980360070757470, 14196301091602199606, 14727918144983470916,
14660430141886012803, 14297932370981794491, 15550237822687034067, 16044915679164358049
], dtype=np.uint64)

dimerTab = np.array([
5015898201438948509, 5225361804584821669, 6423762225589857229, 5783394398799547583,
6894017875502584557, 5959461383092338133, 4833978511655400893, 5364573296520205007,
9002561594443973180, 8212239310050454788, 6941810030513055084, 7579897184553533982,
7935738758488558809, 7149836515649299425, 8257540373175577481, 8935100007508790523
], dtype=np.uint64)

seedA = uint64(0x3c8bfbb395c60474)
seedC = uint64(0x3193c18562a02b4c)
seedG = uint64(0x20323ed082572324)
seedT = uint64(0x295549f54be24456)
seedN = uint64(0x0000000000000000)

seedTab = np.array([
    seedN, seedT, seedN, seedG, seedA, seedA, seedN, seedC, # 0..7
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 8..15
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 16..23
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 24..31
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 32..39
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 40..47
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 48..55
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 56..63
    seedN, seedA, seedN, seedC, seedN, seedN, seedN, seedG, # 64..71
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 72..79
    seedN, seedN, seedN, seedN, seedT, seedT, seedN, seedN, # 80..87
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 88..95
    seedN, seedA, seedN, seedC, seedN, seedN, seedN, seedG, # 96..103
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 104..111
    seedN, seedN, seedN, seedN, seedT, seedT, seedN, seedN, # 112..119
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 120..127
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 128..135
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 136..143
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 144..151
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 152..159
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 160..167
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 168..175
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 176..183
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 184..191
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 192..199
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 200..207
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 208..215
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 216..223
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 224..231
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 232..239
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN, # 240..247
    seedN, seedN, seedN, seedN, seedN, seedN, seedN, seedN  # 248..255
], dtype=np.uint64)

@njit(locals=dict(v=uint64, x=uint64))
def rolx(v, x):
    return (v << x) | (v >> (64 - x))

@njit(locals=dict(v=uint64, x=uint64, y=uint64))
def swapxbits033(v, x):
    y = (v ^ (v >> 33)) & (2**x-1)
    return v ^ (y | (y << 33))

@njit(locals=dict(key=uint64, k=uint64, hVal=uint64,
    currOffSet=uint8, tetramerLoc=uint64, remainder=uint64,
    trimerLoc=uint8, dimerLoc=uint8))
def NT64(key, k):
    hVal=0
    for i in range(int(ceil(k/4))):
        hVal = rolx(hVal, 4)
        hVal = swapxbits033(hVal, 4)
        currOffSet = 4 * i
        tetramerLoc = 64 * ((key>>((currOffSet + 0)*2))&3)\
                    + 16 * ((key>>((currOffSet + 1)*2))&3)\
                    +  4 * ((key>>((currOffSet + 2)*2))&3)\
                    +      ((key>>((currOffSet + 3)*2))&3)
        hVal ^= tetramerTab[tetramerLoc]

    remainder = k % 4
    hVal = rolx(hVal, remainder)
    hVal = swapxbits033(hVal, remainder)

    if remainder == 3:
        trimerLoc = 16 * ((key>>((k-3)*2))&3)\
                  +  4 * ((key>>((k-2)*2))&3)\
                  +      ((key>>((k-1)*2))&3)
        hVal ^= trimerTab[trimerLoc]
    elif remainder == 2:
        dimerLoc = 4 * ((key>>((k-2)*2))&3)\
                 +     ((key>>((k-1)*2))&3)
        hVal ^= dimerTab[dimerLoc]
    elif remainder == 1:
        hVal ^= seedTab[(key>>((k-1)*2))&3]

    return hVal