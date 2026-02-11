import mcdc, math
import numpy as np

# ======================================================================================
# Materials
# ======================================================================================

# Material name: Helium
# ID: 1
# Volume: 1.0
m_helium = mcdc.material(
	nuclides=[
		['He3',4.808864272483583e-10],
		['He4',0.00024044273273775193],
	]
)
# Material name: Inconel
# ID: 3
# Volume: 1.0
m_inconel = mcdc.material(
	nuclides=[
		['Si28',0.0005675748458998167],
		['Si29',2.881983126575598e-05],
		['Si30',1.8998161560800606e-05],
		['Cr50',0.0007823874015570459],
		['Cr52',0.015087561698097418],
		['Cr53',0.0017108083743585282],
		['Cr54',0.00042585641630822467],
		['Mn55',0.0007820073144981398],
		['Fe54',0.0014797392701973641],
		['Fe56',0.02322874201694273],
		['Fe57',0.0005364529531388864],
		['Fe58',7.139204007422978e-05],
		['Ni58',0.02931978281911652],
		['Ni60',0.011293927856516],
		['Ni61',0.0004909392240540197],
		['Ni62',0.0015653290762693152],
		['Ni64',0.00039864316797716485],
	]
)
# Material name: SS302
# ID: 4
# Volume: 1.0
m_ss302 = mcdc.material(
	nuclides=[
		['Si28',0.0015544035849381745],
		['Si29',7.892817900656517e-05],
		['Si30',5.202980831633897e-05],
		['Cr50',0.000711974950938374],
		['Cr52',0.013729727726193934],
		['Cr53',0.0015568409025692384],
		['Cr54',0.00038753065361793535],
		['Mn55',0.0017231784390118814],
		['Fe54',0.003467932096055604],
		['Fe56',0.054439117494534534],
		['Fe57',0.0012572366305896735],
		['Fe58',0.00016731511568472964],
		['Ni58',0.004941136170422727],
		['Ni60',0.0019033168077081128],
		['Ni61',8.273586378242074e-05],
		['Ni62',0.0002637981381064599],
		['Ni64',6.718160869525927e-05],
	]
)
# Material name: SS304
# ID: 5
# Volume: 1.0
m_ss304 = mcdc.material(
	nuclides=[
		['Si28',0.000952813800538438],
		['Si29',4.838116621547466e-05],
		['Si30',3.189308097558792e-05],
		['Cr50',0.0007677835613844196],
		['Cr52',0.014805941187343848],
		['Cr53',0.0016788748692747287],
		['Cr54',0.000417907490970373],
		['Mn55',0.0017604482016877105],
		['Fe54',0.0034619568150176883],
		['Fe56',0.05434531836079257],
		['Fe57',0.0012550703995358764],
		['Fe58',0.0001670268300982717],
		['Ni58',0.005608895030886979],
		['Ni60',0.002160536325402338],
		['Ni61',9.391701811886321e-05],
		['Ni62',0.0002994485508898603],
		['Ni64',7.626071781494654e-05],
	]
)
# Material name: Carbon Steel
# ID: 6
# Volume: 1.0
m_csteel = mcdc.material(
	nuclides=[
		['C12',0.0010442103094126405],
		['C13',1.1697344995776637e-05],
		['Mn55',0.0006412591519223605],
		['P31',3.7913297334043665e-05],
		['S32',3.480801497243693e-05],
		['S33',2.742025443981405e-07],
		['S34',1.536752373542364e-06],
		['S36',5.339824353827618e-09],
		['Si28',0.000617015163486909],
		['Si29',3.1330269529323576e-05],
		['Si30',2.0653053682821572e-05],
		['Ni58',0.0004086181311418285],
		['Ni60',0.00015739897264761742],
		['Ni61',6.842024358597134e-06],
		['Ni62',2.1815367655114368e-05],
		['Ni64',5.555730998971823e-06],
		['Cr50',1.373827831539271e-05],
		['Cr52',0.00026492901252833927],
		['Cr53',3.004082318358793e-05],
		['Cr54',7.477796751321481e-06],
		['Mo92',4.4822291310606895e-05],
		['Mo94',2.810993160790809e-05],
		['Mo95',4.856742618745322e-05],
		['Mo96',5.1015226914058824e-05],
		['Mo97',2.9318533259312206e-05],
		['Mo98',7.432746922042984e-05],
		['Mo100',2.9814212142465947e-05],
		['V50',1.1526145240679119e-07],
		['V51',4.5989319455791894e-05],
		['Nb93',5.055917738603357e-06],
		['Cu63',0.00010223019587827225],
		['Cu65',4.56081209267397e-05],
		['Ca40',1.704268333890243e-05],
		['Ca42',1.1374564024217834e-07],
		['Ca43',2.3733634358428184e-08],
		['Ca44',3.6672860192375593e-07],
		['Ca46',7.032187772753565e-10],
		['Ca48',3.287547867637643e-08],
		['B10',2.5832795133853293e-06],
		['B11',1.0450421269107751e-05],
		['Ti46',1.214386179605987e-06],
		['Ti47',1.0951555387252844e-06],
		['Ti48',1.0851460531518844e-05],
		['Ti49',7.963429398385609e-07],
		['Ti50',7.624873257424189e-07],
		['Al27',4.352299802675485e-05],
		['Fe54',0.004743658379473851],
		['Fe56',0.07446529191498036],
		['Fe57',0.0017197283316076073],
		['Fe58',0.0002288642708527182],
	]
)
# Material name: Zircaloy-4
# ID: 7
# Volume: 1.0
m_zr4 = mcdc.material(
	nuclides=[
		['O16',0.00030744435226246966],
		['O17',1.1679932181228452e-07],
		['O18',6.164785312704674e-07],
		['Cr50',3.296180328418399e-06],
		['Cr52',6.356355428793489e-05],
		['Cr53',7.207596771153883e-06],
		['Cr54',1.7941233963793304e-06],
		['Fe54',8.669830240139012e-06],
		['Fe56',0.00013609779373633635],
		['Fe57',3.143091576474185e-06],
		['Fe58',4.1828778921182414e-07],
		['Zr90',0.02182757976935886],
		['Zr91',0.004760066934270842],
		['Zr92',0.007275859888090276],
		['Zr94',0.00737343710204759],
		['Zr96',0.0011878955606900586],
		['Sn112',4.673521272730707e-06],
		['Sn114',3.1799215898767953e-06],
		['Sn115',1.6381414619925706e-06],
		['Sn116',7.005463783334945e-05],
		['Sn117',3.7002724807551606e-05],
		['Sn118',0.00011669348891480064],
		['Sn119',4.138716226578583e-05],
		['Sn120',0.00015697249778294243],
		['Sn122',2.230763257260888e-05],
		['Sn124',2.789658616579101e-05],
	]
)
# Material name: m_ss304
# ID: 8
# Volume: 1.0
m_m5 = mcdc.material(
	nuclides=[
		['Zr90',0.021826659699624183],
		['Zr91',0.004759866313504049],
		['Zr92',0.007275553233208061],
		['Zr94',0.007373126250329802],
		['Zr96',0.0011878454258298875],
		['Nb93',0.00042910080334290177],
		['O16',5.7790773120342736e-05],
		['O17',2.195494260303957e-08],
		['O18',1.15880388345964e-07],
	]
)
# Material name: Borated Water
# ID: 10
# Volume: 1.0
# S(a,b): c_H_in_H2O
m_water = mcdc.material(
	nuclides=[
		['B10',1.0323440206972448e-05],
		['B11',4.1762534601163005e-05],
		['H1',0.050347844752850625],
		['H2',7.842394716362082e-06],
		['O16',0.025117935412784034],
		['O17',9.542402714463945e-06],
		['O18',5.03657582849965e-05],
	]
)
# Material name: 3.1% Enr. UO2 Fuel
# ID: 18
# Volume: 103.4494453869801
# Depletable: {depletable}
m_fuel31 = mcdc.material(
	nuclides=[
		#['U234',5.79887580805e-06],
		['U235',0.000721754675808],
		#['U236',2.8985972088e-09],
		#['U237',3.07856784763e-11],
		['U238',0.0222540811763],
		#['U239',1.24589215921e-11],
		#['U240',2.0545991913e-18],
		#['Np236',3.42433788929e-21],
		['Np237',1.42910567478e-11],
		['Np238',1.66404828937e-17],
		#['Np239',1.70341590939e-09],
		#['Pu236',1.09076957926e-29],
		#['Pu237',3.06080855723e-28],
		['Pu238',1.3223107364e-17],
		['Pu239',4.06606871885e-09],
		['Pu240',2.99149434985e-14],
		#['Pu241',5.98896912331e-19],
		['Pu242',1.22264626294e-24],
		['B11',1.40629256107e-27],
		['N14',6.51626007099e-19],
		['N15',5.18607979214e-28],
		['O16',0.0459458933267],
		['O17',1.74200967393e-05],
		['F19',4.89885015203e-31],
		['Fe57',3.94025927987e-57],
		['Fe58',5.9044332143e-50],
		['Co59',5.54702778074e-57],
		['Ni60',2.11578225359e-46],
		['Ni61',2.15262487282e-39],
		['Ni62',9.0497961839e-44],
		['Cu63',1.74057869502e-32],
		['Ni64',3.53079707081e-27],
		['Zn64',1.86683623254e-40],
		['Cu65',8.50771501029e-32],
		['Zn65',1.11533310555e-48],
		['Zn66',3.72341594147e-18],
		['Zn67',2.7516037089e-17],
		['Zn68',1.36384091597e-20],
		['Ga69',1.75512362258e-16],
		['Zn70',4.87601660885e-16],
		['Ge70',1.29697741737e-20],
		['Ga71',1.95764881272e-15],
		['Ge72',2.39607689684e-15],
		['Ge73',1.28723371295e-14],
		['Ge74',4.33915594763e-14],
		['As74',2.60900896856e-21],
		['Se74',2.25609045791e-22],
		['As75',1.35604908492e-13],
		['Ge76',4.08284733105e-13],
		['Se76',2.08748011652e-17],
	]
)
# Material name: 2.4% Enr. UO2 Fuel
# ID: 19
# Volume: 103.4494453869801
# Depletable: {depletable}
m_fuel24 = mcdc.material(
	nuclides=[
		#['U234',4.48429833983e-06],
		['U235',0.000558129874969],
		#['U236',4.39014521603e-09],
		#['U237',4.00720948365e-11],
		['U238',0.0224078395611],
		#['U239',2.14649505842e-11],
		#['U240',4.61060281201e-18],
		#['Np234',2.55400862099e-43],
		#['Np235',4.8424193802e-31],
		#['Np236',3.97728091346e-21],
		['Np237',3.72803230839e-11],
		['Np238',1.21121656937e-16],
		#['Np239',3.34921887205e-09],
		#['Pu236',3.41033180932e-29],
		#['Pu237',8.26790363453e-30],
		['Pu238',2.34172378623e-16],
		['Pu239',1.00552294268e-08],
		['Pu240',1.3886306488e-13],
		#['Pu241',2.20764706334e-18],
		['Pu242',9.1348447872e-24],
		['B11',5.66882255592e-27],
		['N14',1.54824894659e-18],
		['N15',2.2608298436e-27],
		['O16',0.0459235755974],
		['O17',1.74116359401e-05],
		['Fe57',5.57126412519e-57],
		['Fe58',6.43781127667e-50],
		['Co59',6.3404259543e-58],
		['Ni60',2.8877355906e-46],
		['Ni61',1.74612711314e-39],
		['Ni62',3.67783085548e-44],
		['Cu63',4.03301497835e-32],
		['Ni64',1.09232867403e-26],
		['Zn64',6.99543081223e-40],
		['Cu65',1.63711332667e-31],
		['Zn65',7.82374309956e-48],
		['Zn66',6.82234230984e-18],
		['Zn67',4.92836528559e-17],
		['Zn68',2.59548074295e-20],
		['Ga69',2.97159881684e-16],
		['Zn70',8.29775150294e-16],
		['Ge70',2.64737950649e-20],
		['Ga71',3.38117418187e-15],
		['Ge72',4.31830046359e-15],
		['Ge73',2.1405857243e-14],
		['Ge74',7.13221952382e-14],
		['As74',4.61436701203e-21],
		['Se74',4.41611226462e-22],
		['As75',2.23308411105e-13],
		['Ge76',6.71115528762e-13],
	]
)
# Material name: 1.6% Enr. UO2 Fuel
# ID: 20
# Volume: 103.4494453869801
# Depletable: {depletable}
m_fuel16 = mcdc.material(
	nuclides=[
		#['U234',3.01290682194e-06],
		['U235',0.00037495034066],
		#['U236',1.45737226331e-08],
		#['U237',1.06320422099e-10],
		['U238',0.0226258447241],
		#['U239',1.25432816286e-10],
		#['U240',3.09667016082e-16],
		#['Np234',5.6700725953e-41],
		#['Np235',7.68764769087e-30],
		#['Np236',1.56000462831e-20],
		['Np237',6.46555701352e-11],
		['Np238',1.18963869973e-15],
		#['Np239',1.6486353533e-08],
		#['Pu236',9.09840861647e-29],
		#['Pu237',1.02426625479e-26],
		['Pu238',1.29895845159e-15],
		['Pu239',3.95827233885e-08],
		['Pu240',3.0671432531e-12],
		#['Pu241',3.98036491214e-16],
		['Pu242',1.07092391805e-20],
		['B11',1.75989269508e-25],
		['N14',6.27000581377e-18],
		['N15',5.73935209994e-26],
		['O16',0.0459904787404],
		['O17',1.74370070382e-05],
		['Fe57',3.87623539477e-54],
		['Fe58',1.10547983072e-46],
		['Co59',2.10035858561e-55],
		['Ni60',2.5123675585e-44],
		['Ni61',2.96838403021e-37],
		['Ni62',2.72054655342e-42],
		['Cu63',5.85991814778e-31],
		['Ni64',1.91590249532e-25],
		['Zn64',5.53245052637e-38],
		['Cu65',2.43090804717e-30],
		['Zn65',2.95249411477e-45],
		['Zn66',2.37643846769e-17],
		['Zn67',1.66182953626e-16],
		['Zn68',1.10797609041e-19],
		['Ga69',1.09539280488e-15],
		['Zn70',3.09290296573e-15],
		['Ge70',1.22646482967e-19],
		['Ga71',1.28564587943e-14],
		['Ge72',1.42025928411e-14],
		['Ge73',7.56700585887e-14],
		['Ge74',2.52575350246e-13],
		['As74',1.94337160439e-20],
		['Se74',1.59240348107e-21],
		['As75',7.89006840569e-13],
		['Ge76',2.37810681716e-12],
	]
)
# Material name: Ag-In-Cd
# ID: 21
# Volume: -
# Depletable: -
m_cr = mcdc.material(
    nuclides=[
        ['Ag107', 0.023523285675833942],
        ['Ag109', 0.02185429814297804],
        ['In113', 0.0003421922042655644],
        ['In115', 0.007651085167039375],
        ['Cd106', 3.38816276451386e-05],
        ['Cd108', 2.4166172970990425e-05],
        ['Cd110', 0.0003393605596264083],
        ['Cd111', 0.0003482051612205208],
        ['Cd112', 0.0006561061533306398],
        ['Cd113', 0.00033274751904988726],
        ['Cd114', 0.0007825159207295705],
        ['Cd116', 0.00020443276053837845],
    ]
)


# ======================================================================================
# Geometry
# ======================================================================================

# --------------------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------------------

s1 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.405765) # Name: Pellet OR
s2 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.06459) # Name: FR Plenum Spring OR
s3 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.41402) # Name: Clad IR
s4 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.47498) # Name: Clad OR
s5 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.5715) # Name: GT IR (above dashpot)
s6 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.61214) # Name: GT OR (above dashpot)
s7 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.50419) # Name: GT IR (at dashpot)
s8 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=0.61214) # Name: GT OR (at dashpot)
s20 = mcdc.surface('plane-x', x=-0.62208) # Name: minimum x
s21 = mcdc.surface('plane-x', x=0.62208) # Name: maximum x
s22 = mcdc.surface('plane-y', y=-0.62208) # Name: minimum y
s23 = mcdc.surface('plane-y', y=0.62208) # Name: maximum y
s24 = mcdc.surface('plane-x', x=-10.70864) # Name: minimum x
s25 = mcdc.surface('plane-x', x=10.70864) # Name: maximum x
s26 = mcdc.surface('plane-y', y=-10.70864) # Name: minimum y
s27 = mcdc.surface('plane-y', y=10.70864) # Name: maximum y
s28 = mcdc.surface('plane-x', x=-10.73635) # Name: minimum x
s29 = mcdc.surface('plane-x', x=10.73635) # Name: maximum x
s30 = mcdc.surface('plane-y', y=-10.73635) # Name: minimum y
s31 = mcdc.surface('plane-y', y=10.73635) # Name: maximum y
s32 = mcdc.surface('plane-z', z=-16.6205) # Name: bot support plate
s33 = mcdc.surface('plane-z', z=-11.6205) # Name: top support plate
s34 = mcdc.surface('plane-z', z=-1.4604999999999997) # Name: bottom FR
s35 = mcdc.surface('plane-z', z=2.220446049250313e-16) # Name: bot active core
s36 = mcdc.surface('plane-z', z=199.9996) # Name: top active core
s38 = mcdc.surface('plane-z', z=4.5395) # Name: bottom grid 1
s39 = mcdc.surface('plane-z', z=8.9845) # Name: top of grid 1
s40 = mcdc.surface('plane-z', z=55.40325) # Name: bottom grid 2
s41 = mcdc.surface('plane-z', z=59.84825) # Name: top of grid 2
s42 = mcdc.surface('plane-z', z=106.26700000000001) # Name: bottom grid 3
s43 = mcdc.surface('plane-z', z=110.71200000000002) # Name: top of grid 3
s44 = mcdc.surface('plane-z', z=157.13075) # Name: bottom grid 4
s45 = mcdc.surface('plane-z', z=161.57575) # Name: top of grid 4
s46 = mcdc.surface('plane-z', z=207.99450000000002) # Name: bottom grid 5
s47 = mcdc.surface('plane-z', z=212.4395) # Name: top of grid 5
s48 = mcdc.surface('plane-z', z=46.079) # Name: top dashpot
s49 = mcdc.surface('plane-z', z=213.48953999999998) # Name: top pin plenum
s50 = mcdc.surface('plane-z', z=214.4395) # Name: top FR
s51 = mcdc.surface('plane-z', z=217.78449999999998) # Name: bottom upper nozzle
s52 = mcdc.surface('plane-z', z=226.61149999999998) # Name: top upper nozzle
s71 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=93.98) # Name: core barrel IR
s72 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=99.06) # Name: core barrel OR
s78 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=122.555) # Name: RPV IR
s79 = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=133.35, bc='vacuum') # Name: RPV OR
s80 = mcdc.surface('plane-z', z=246.61149999999998, bc='vacuum') # Name: upper problem boundary
s81 = mcdc.surface('plane-z', z=-36.6205, bc='vacuum') # Name: lower problem boundary
s82 = mcdc.surface('cylinder-z', center=[-9.816879130434781, -6.155027391304348], radius=0.506426304347826)
s83 = mcdc.surface('cylinder-z', center=[-4.518880869565217, -6.155027391304348], radius=0.506426304347826)
s84 = mcdc.surface('cylinder-z', center=[0.9349408695652173, -6.155027391304348], radius=0.506426304347826)
s85 = mcdc.surface('cylinder-z', center=[6.155027391304348, -6.155027391304348], radius=1.1686760869565216)
s86 = mcdc.surface('cylinder-z', center=[6.155027391304348, -0.9349408695652173], radius=0.506426304347826)
s87 = mcdc.surface('cylinder-z', center=[6.155027391304348, 4.518880869565216], radius=0.506426304347826)
s88 = mcdc.surface('cylinder-z', center=[6.155027391304348, 9.816879130434781], radius=0.506426304347826)
s89 = mcdc.surface('cylinder-z', center=[2.5710873913043475, -2.5710873913043475], radius=0.506426304347826)
s90 = mcdc.surface('cylinder-z', center=[-2.0257052173913053, -1.246587826086957], radius=0.506426304347826)
s91 = mcdc.surface('cylinder-z', center=[1.2465878260869552, 2.0257052173913035], radius=0.506426304347826)
s92 = mcdc.surface('cylinder-z', center=[-6.544586086956521, 0.0], radius=0.506426304347826)
s93 = mcdc.surface('cylinder-z', center=[0.0, 6.544586086956521], radius=0.506426304347826)
s94 = mcdc.surface('cylinder-z', center=[-9.816879130434781, 8.49237956521739], radius=0.506426304347826)
s95 = mcdc.surface('cylinder-z', center=[6.155027391304348, -6.155027391304348], radius=0.506426304347826)
s96 = mcdc.surface('cylinder-z', center=[2.726910869565218, 1.402411304347826], radius=0.506426304347826)
s97 = mcdc.surface('cylinder-z', center=[-1.5582347826086949, -2.726910869565218], radius=0.506426304347826)
s98 = mcdc.surface('cylinder-z', center=[6.232939130434779, 0.0], radius=0.506426304347826)
s99 = mcdc.surface('cylinder-z', center=[6.232939130434779, 5.2200865217391295], radius=0.506426304347826)
s100 = mcdc.surface('cylinder-z', center=[6.232939130434779, 10.440173043478259], radius=0.506426304347826)
s101 = mcdc.surface('cylinder-z', center=[6.232939130434779, -5.2200865217391295], radius=0.506426304347826)
s102 = mcdc.surface('cylinder-z', center=[6.232939130434779, -10.440173043478259], radius=0.506426304347826)
s103 = mcdc.surface('cylinder-z', center=[1.5582347826086949, 2.6100432608695647], radius=0.506426304347826)
s104 = mcdc.surface('cylinder-z', center=[1.5582347826086949, 7.830129782608694], radius=0.506426304347826)
s105 = mcdc.surface('cylinder-z', center=[1.5582347826086949, -2.6100432608695647], radius=0.506426304347826)
s106 = mcdc.surface('cylinder-z', center=[1.5582347826086949, -7.830129782608694], radius=0.506426304347826)
s107 = mcdc.surface('cylinder-z', center=[-2.726910869565218, 5.921292173913042], radius=0.506426304347826)
s108 = mcdc.surface('cylinder-z', center=[-2.726910869565218, -5.921292173913042], radius=0.506426304347826)
s109 = mcdc.surface('cylinder-z', center=[6.232939130434779, -11.063466956521738], radius=0.506426304347826)
s110 = mcdc.surface('cylinder-z', center=[6.232939130434779, -5.84338043478261], radius=0.506426304347826)
s111 = mcdc.surface('cylinder-z', center=[6.232939130434779, -0.6232939130434794], radius=0.506426304347826)
s112 = mcdc.surface('cylinder-z', center=[6.232939130434779, 4.596792608695651], radius=0.506426304347826)
s113 = mcdc.surface('cylinder-z', center=[6.232939130434779, 9.816879130434778], radius=0.506426304347826)
s114 = mcdc.surface('cylinder-z', center=[1.5582347826086949, -8.453423695652173], radius=0.506426304347826)
s115 = mcdc.surface('cylinder-z', center=[1.5582347826086949, -3.2333371739130428], radius=0.506426304347826)
s116 = mcdc.surface('cylinder-z', center=[1.5582347826086949, 7.206835869565214], radius=0.506426304347826)
s117 = mcdc.surface('cylinder-z', center=[-0.15582347826087073, 1.6361465217391302], radius=0.506426304347826)
s118 = mcdc.surface('cylinder-z', center=[-3.5060282608695648, -7.089968260869565], radius=0.506426304347826)
s119 = mcdc.surface('cylinder-z', center=[6.232939130434779, 11.063466956521738], radius=0.506426304347826)
s120 = mcdc.surface('cylinder-z', center=[6.232939130434779, 5.84338043478261], radius=0.506426304347826)
s121 = mcdc.surface('cylinder-z', center=[6.232939130434779, 0.6232939130434794], radius=0.506426304347826)
s122 = mcdc.surface('cylinder-z', center=[6.232939130434779, -4.596792608695651], radius=0.506426304347826)
s123 = mcdc.surface('cylinder-z', center=[6.232939130434779, -9.816879130434778], radius=0.506426304347826)
s124 = mcdc.surface('cylinder-z', center=[1.5582347826086949, 8.453423695652173], radius=0.506426304347826)
s125 = mcdc.surface('cylinder-z', center=[1.5582347826086949, 3.2333371739130428], radius=0.506426304347826)
s126 = mcdc.surface('cylinder-z', center=[1.5582347826086949, -7.206835869565214], radius=0.506426304347826)
s127 = mcdc.surface('cylinder-z', center=[-0.15582347826087073, -1.6361465217391302], radius=0.506426304347826)
s128 = mcdc.surface('cylinder-z', center=[-3.5060282608695648, 7.089968260869565], radius=0.506426304347826)
s129 = mcdc.surface('cylinder-z', center=[4.674704347826085, -3.1164695652173915], radius=0.506426304347826)

# --------------------------------------------------------------------------------------
# Moving control rods
# --------------------------------------------------------------------------------------

config = {}

# Design spec.
config['top'] = 199.9996
config['bottom'] = 2.220446049250313e-16
config['length'] = config['top'] - config['bottom']

# Problem spec.
config['frac_r'] = np.array([1.0, 1.0, 0.35, 1.0])
config['time_r'] = np.array([0.0, 10.0, 15.0, 15.65])
rxf = 0.9
config['frac_rx'] = np.array([1.0, 1.0, rxf, 1.0])
config['time_rx'] = np.array([0.0, 10.0, 15.0, 15.0 + 1.0 - rxf])
config['frac_s'] = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
config['time_s'] = np.array([0.0, 5.0, 10.0, 15.0, 16.0])
config['frac_sx2'] = np.array([0.65, 0.65, 1.0])
config['time_sx2'] = np.array([0.0, 15.0, 15.35])
config['frac_sx3'] = np.array([1.0, 1.0, 0.2, 0.2, 1.0])
config['time_sx3'] = np.array([0.0, 5.0, 9.0, 15.0, 15.8])
cases = ['r', 'rx', 's', 'sx2', 'sx3']

config['frac_r'] = np.array([1.0, 1.0])
config['time_r'] = np.array([0.0, 10.0])

config['frac_rx'] = np.array([0.5, 0.5])
config['time_rx'] = np.array([0.0, 10.0])

config['frac_s'] = np.array([1.0, 1.0, 0.0])
config['time_s'] = np.array([0.0, 5.0, 10.0])

config['frac_sx2'] = np.array([1.0, 1.0, 0.5])
config['time_sx2'] = np.array([0.0, 5.0, 7.5])

config['frac_sx3'] = np.array([0.75, 0.75])
config['time_sx3'] = np.array([0.0, 10.0])

# All cases
for case in cases:
    # The CR tip positions
    z_top = config['top'] + (1.0 - config[f'frac_{case}']) * config['length']
    z_bottom = z_top - config['length']

    # The CR tip surfaces
    s_top = mcdc.surface("plane-z", z=z_top[0])
    s_bottom = mcdc.surface("plane-z", z=z_bottom[0])

    # Tip moving durations and velocities
    t_move = config[f'time_{case}']
    durations = t_move[1:] - t_move[:-1]
    velocities = np.zeros((len(durations), 3))
    velocities[:, 2] = (z_top[1:] - z_top[:-1]) / durations

    # Move the surfaces
    s_top.move(velocities, durations)
    s_bottom.move(velocities, durations)

    # CR region
    cr_region = +s_bottom & -s_top

    # CR and water cells
    cell_cr = mcdc.cell(cr_region, fill=m_cr)
    cell_water = mcdc.cell(~cr_region, fill=m_water)

    # Universe
    univ_cr_water = mcdc.universe([cell_cr, cell_water])

    # Universe cells
    config[f'cell_cr_water_{case}'] = mcdc.cell(fill=univ_cr_water)
    config[f'cell_cr_water_tube_{case}'] = mcdc.cell(-s5, fill=univ_cr_water)
    config[f'cell_cr_water_dashpot_{case}'] = mcdc.cell(-s7, fill=univ_cr_water)

    # Continue to Moving control rods - Level 1

# --------------------------------------------------------------------------------------
# Cells - Level 0
# --------------------------------------------------------------------------------------

c1 = mcdc.cell(fill=m_water) # Name: water pin
c2 = mcdc.cell(-s5, fill=m_water) # Name: GT empty (0)
c3 = mcdc.cell(+s5 & -s6, fill=m_zr4) # Name: GT empty (1)
c4 = mcdc.cell(+s6, fill=m_water) # Name: GT empty (last)
c5 = mcdc.cell(-s5, fill=m_water) # Name: GT empty grid (bottom) (0)
c6 = mcdc.cell(+s5 & -s6, fill=m_zr4) # Name: GT empty grid (bottom) (1)
c7 = mcdc.cell(+s6 & +s20 & -s21 & +s22 & -s23, fill=m_water) # Name: GT empty grid (bottom) (last)
c8 = mcdc.cell(-s20 | +s21 | -s22 | +s23, fill=m_inconel) # Name: GT empty grid (bottom) (grid)
c9 = mcdc.cell(-s5, fill=m_water) # Name: GT empty grid (intermediate) (0)
c10 = mcdc.cell(+s5 & -s6, fill=m_zr4) # Name: GT empty grid (intermediate) (1)
c11 = mcdc.cell(+s6 & +s20 & -s21 & +s22 & -s23, fill=m_water) # Name: GT empty grid (intermediate) (last)
c12 = mcdc.cell(-s20 | +s21 | -s22 | +s23, fill=m_zr4) # Name: GT empty grid (intermediate) (grid)
c16 = mcdc.cell(-s7, fill=m_water) # Name: GT empty at dashpot (0)
c17 = mcdc.cell(+s7 & -s8, fill=m_zr4) # Name: GT empty at dashpot (1)
c18 = mcdc.cell(+s8, fill=m_water) # Name: GT empty at dashpot (last)
c19 = mcdc.cell(-s7, fill=m_water) # Name: GT empty at dashpot grid (bottom) (0)
c20 = mcdc.cell(+s7 & -s8, fill=m_zr4) # Name: GT empty at dashpot grid (bottom) (1)
c21 = mcdc.cell(+s8 & +s20 & -s21 & +s22 & -s23, fill=m_water) # Name: GT empty at dashpot grid (bottom) (last)
c22 = mcdc.cell(-s20 | +s21 | -s22 | +s23, fill=m_inconel) # Name: GT empty at dashpot grid (bottom) (grid)
c585 = mcdc.cell(-s4, fill=m_ss304) # Name: SS pin (0)
c586 = mcdc.cell(+s4, fill=m_water) # Name: SS pin (last)
c587 = mcdc.cell(-s4, fill=m_m5) # Name: end plug (0)
c588 = mcdc.cell(+s4, fill=m_water) # Name: end plug (last)
c589 = mcdc.cell(-s2, fill=m_ss302) # Name: pin plenum (0)
c590 = mcdc.cell(+s2 & -s3, fill=m_helium) # Name: pin plenum (1)
c591 = mcdc.cell(+s3 & -s4, fill=m_m5) # Name: pin plenum (2)
c592 = mcdc.cell(+s4, fill=m_water) # Name: pin plenum (last)
c593 = mcdc.cell(-s2, fill=m_inconel) # Name: pin plenum grid (intermediate) (0)
c594 = mcdc.cell(+s2 & -s3, fill=m_helium) # Name: pin plenum grid (intermediate) (1)
c595 = mcdc.cell(+s3 & -s4, fill=m_zr4) # Name: pin plenum grid (intermediate) (2)
c596 = mcdc.cell(+s4 & +s20 & -s21 & +s22 & -s23, fill=m_water) # Name: pin plenum grid (intermediate) (last)
c597 = mcdc.cell(-s20 | +s21 | -s22 | +s23, fill=m_zr4) # Name: pin plenum grid (intermediate) (grid)
c598 = mcdc.cell(-s3, fill=m_helium) # Name: Outside pin (0)
c599 = mcdc.cell(+s3 & -s4, fill=m_m5) # Name: Outside pin (1)
c600 = mcdc.cell(+s4, fill=m_water) # Name: Outside pin (last)
c601 = mcdc.cell(-s3, fill=m_helium) # Name: Outside pin grid (bottom) (0)
c602 = mcdc.cell(+s3 & -s4, fill=m_m5) # Name: Outside pin grid (bottom) (1)
c603 = mcdc.cell(+s4 & +s20 & -s21 & +s22 & -s23, fill=m_water) # Name: Outside pin grid (bottom) (last)
c604 = mcdc.cell(-s20 | +s21 | -s22 | +s23, fill=m_inconel) # Name: Outside pin grid (bottom) (grid)
c605 = mcdc.cell(-s3, fill=m_helium) # Name: Outside pin grid (intermediate) (0)
c606 = mcdc.cell(+s3 & -s4, fill=m_m5) # Name: Outside pin grid (intermediate) (1)
c607 = mcdc.cell(+s4 & +s20 & -s21 & +s22 & -s23, fill=m_water) # Name: Outside pin grid (intermediate) (last)
c608 = mcdc.cell(-s20 | +s21 | -s22 | +s23, fill=m_zr4) # Name: Outside pin grid (intermediate) (grid)
c623 = mcdc.cell(-s1, fill=m_fuel16) # Name: Fuel pin (1.6%) stack (i)
c650 = mcdc.cell(-s1, fill=m_fuel24) # Name: Fuel pin (2.4%) stack (i)
c677 = mcdc.cell(-s1, fill=m_fuel31) # Name: Fuel pin (3.1%) stack (i)
c691 = mcdc.cell(-s28 | +s29 | -s30 | +s31, fill=m_water) # Name: Assembly (1.6%) no BAs lattice outer water
c692 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & -s38, fill=m_water) # Name: Assembly (1.6%) no BAs lattice axial (0)
c693 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s38 & -s39, fill=m_inconel) # Name: Assembly (1.6%) no BAs lattice axial (1)
c694 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s39 & -s40, fill=m_water) # Name: Assembly (1.6%) no BAs lattice axial (2)
c695 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s40 & -s41, fill=m_zr4) # Name: Assembly (1.6%) no BAs lattice axial (3)
c696 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s41 & -s42, fill=m_water) # Name: Assembly (1.6%) no BAs lattice axial (4)
c697 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s42 & -s43, fill=m_zr4) # Name: Assembly (1.6%) no BAs lattice axial (5)
c698 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s43 & -s44, fill=m_water) # Name: Assembly (1.6%) no BAs lattice axial (6)
c699 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s44 & -s45, fill=m_zr4) # Name: Assembly (1.6%) no BAs lattice axial (7)
c700 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s45 & -s46, fill=m_water) # Name: Assembly (1.6%) no BAs lattice axial (8)
c701 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s46 & -s47, fill=m_zr4) # Name: Assembly (1.6%) no BAs lattice axial (9)
c702 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s47, fill=m_water) # Name: Assembly (1.6%) no BAs lattice axial (last)
c925 = mcdc.cell(-s28 | +s29 | -s30 | +s31, fill=m_water) # Name: Assembly (2.4%) no BAs lattice outer water
c926 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & -s38, fill=m_water) # Name: Assembly (2.4%) no BAs lattice axial (0)
c927 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s38 & -s39, fill=m_inconel) # Name: Assembly (2.4%) no BAs lattice axial (1)
c928 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s39 & -s40, fill=m_water) # Name: Assembly (2.4%) no BAs lattice axial (2)
c929 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s40 & -s41, fill=m_zr4) # Name: Assembly (2.4%) no BAs lattice axial (3)
c930 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s41 & -s42, fill=m_water) # Name: Assembly (2.4%) no BAs lattice axial (4)
c931 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s42 & -s43, fill=m_zr4) # Name: Assembly (2.4%) no BAs lattice axial (5)
c932 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s43 & -s44, fill=m_water) # Name: Assembly (2.4%) no BAs lattice axial (6)
c933 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s44 & -s45, fill=m_zr4) # Name: Assembly (2.4%) no BAs lattice axial (7)
c934 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s45 & -s46, fill=m_water) # Name: Assembly (2.4%) no BAs lattice axial (8)
c935 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s46 & -s47, fill=m_zr4) # Name: Assembly (2.4%) no BAs lattice axial (9)
c936 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s47, fill=m_water) # Name: Assembly (2.4%) no BAs lattice axial (last)
c977 = mcdc.cell(-s28 | +s29 | -s30 | +s31, fill=m_water) # Name: Assembly (3.1%) no BAs lattice outer water
c978 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & -s38, fill=m_water) # Name: Assembly (3.1%) no BAs lattice axial (0)
c979 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s38 & -s39, fill=m_inconel) # Name: Assembly (3.1%) no BAs lattice axial (1)
c980 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s39 & -s40, fill=m_water) # Name: Assembly (3.1%) no BAs lattice axial (2)
c981 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s40 & -s41, fill=m_zr4) # Name: Assembly (3.1%) no BAs lattice axial (3)
c982 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s41 & -s42, fill=m_water) # Name: Assembly (3.1%) no BAs lattice axial (4)
c983 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s42 & -s43, fill=m_zr4) # Name: Assembly (3.1%) no BAs lattice axial (5)
c984 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s43 & -s44, fill=m_water) # Name: Assembly (3.1%) no BAs lattice axial (6)
c985 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s44 & -s45, fill=m_zr4) # Name: Assembly (3.1%) no BAs lattice axial (7)
c986 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s45 & -s46, fill=m_water) # Name: Assembly (3.1%) no BAs lattice axial (8)
c987 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s46 & -s47, fill=m_zr4) # Name: Assembly (3.1%) no BAs lattice axial (9)
c988 = mcdc.cell(+s28 & -s29 & +s30 & -s31 & (-s24 | +s25 | -s26 | +s27) & +s47, fill=m_water) # Name: Assembly (3.1%) no BAs lattice axial (last)
c1028 = mcdc.cell(-s82, fill=m_water)
c1029 = mcdc.cell(-s83, fill=m_water)
c1030 = mcdc.cell(-s84, fill=m_water)
c1031 = mcdc.cell(-s85, fill=m_water)
c1032 = mcdc.cell(-s86, fill=m_water)
c1033 = mcdc.cell(-s87, fill=m_water)
c1034 = mcdc.cell(-s88, fill=m_water)
c1035 = mcdc.cell(-s89, fill=m_water)
c1036 = mcdc.cell(-s90, fill=m_water)
c1037 = mcdc.cell(-s91, fill=m_water)
c1038 = mcdc.cell(-s92, fill=m_water)
c1039 = mcdc.cell(-s93, fill=m_water)
c1040 = mcdc.cell(-s94, fill=m_water)
c1041 = mcdc.cell(+s82 & +s83 & +s84 & +s85 & +s86 & +s87 & +s88 & +s89 & +s90 & +s91 & +s92 & +s93 & +s94, fill=m_ss304) # Name: reflector NW SS
c1042 = mcdc.cell(-s95, fill=m_water)
c1043 = mcdc.cell(-s96, fill=m_water)
c1044 = mcdc.cell(-s97, fill=m_water)
c1045 = mcdc.cell(+s95 & +s96 & +s97, fill=m_ss304) # Name: reflector 1,1 SS
c1046 = mcdc.cell(-s98, fill=m_water)
c1047 = mcdc.cell(-s99, fill=m_water)
c1048 = mcdc.cell(-s100, fill=m_water)
c1049 = mcdc.cell(-s101, fill=m_water)
c1050 = mcdc.cell(-s102, fill=m_water)
c1051 = mcdc.cell(-s103, fill=m_water)
c1052 = mcdc.cell(-s104, fill=m_water)
c1053 = mcdc.cell(-s105, fill=m_water)
c1054 = mcdc.cell(-s106, fill=m_water)
c1055 = mcdc.cell(-s107, fill=m_water)
c1056 = mcdc.cell(-s108, fill=m_water)
c1057 = mcdc.cell(+s98 & +s99 & +s100 & +s101 & +s102 & +s103 & +s104 & +s105 & +s106 & +s107 & +s108, fill=m_ss304) # Name: reflector 4,0 SS
c1058 = mcdc.cell(-s109, fill=m_water)
c1059 = mcdc.cell(-s110, fill=m_water)
c1060 = mcdc.cell(-s111, fill=m_water)
c1061 = mcdc.cell(-s112, fill=m_water)
c1062 = mcdc.cell(-s113, fill=m_water)
c1063 = mcdc.cell(-s114, fill=m_water)
c1064 = mcdc.cell(-s115, fill=m_water)
c1065 = mcdc.cell(-s116, fill=m_water)
c1066 = mcdc.cell(-s117, fill=m_water)
c1067 = mcdc.cell(-s118, fill=m_water)
c1068 = mcdc.cell(+s109 & +s110 & +s111 & +s112 & +s113 & +s114 & +s115 & +s116 & +s117 & +s118, fill=m_ss304) # Name: reflector 3,0 SS
c1069 = mcdc.cell(-s119, fill=m_water)
c1070 = mcdc.cell(-s120, fill=m_water)
c1071 = mcdc.cell(-s121, fill=m_water)
c1072 = mcdc.cell(-s122, fill=m_water)
c1073 = mcdc.cell(-s123, fill=m_water)
c1074 = mcdc.cell(-s124, fill=m_water)
c1075 = mcdc.cell(-s125, fill=m_water)
c1076 = mcdc.cell(-s126, fill=m_water)
c1077 = mcdc.cell(-s127, fill=m_water)
c1078 = mcdc.cell(-s128, fill=m_water)
c1079 = mcdc.cell(+s119 & +s120 & +s121 & +s122 & +s123 & +s124 & +s125 & +s126 & +s127 & +s128, fill=m_ss304) # Name: reflector 5,0 SS
c1080 = mcdc.cell(-s129, fill=m_water)
c1081 = mcdc.cell(+s129, fill=m_ss304) # Name: reflector 2,0 SS
c1104 = mcdc.cell(fill=m_ss304) # Name: heavy reflector
c1106 = mcdc.cell(+s71 & -s72 & +s81 & -s80, fill=m_ss304) # Name: core barrel
c1107 = mcdc.cell(+s72 & -s78 & +s81 & -s80, fill=m_water) # Name: downcomer
c1108 = mcdc.cell(+s78 & -s79 & +s81 & -s80, fill=m_csteel) # Name: reactor pressure vessel

# --------------------------------------------------------------------------------------
# Universes - Level 1
# --------------------------------------------------------------------------------------

u1 = mcdc.universe([c1])
u150 = mcdc.universe([c1028, c1029, c1030, c1031, c1032, c1033, c1034, c1035, c1036, c1037, c1038, c1039, c1040, c1041])
u151 = mcdc.universe([c1042, c1043, c1044, c1045])
u152 = mcdc.universe([c1046, c1047, c1048, c1049, c1050, c1051, c1052, c1053, c1054, c1055, c1056, c1057])
u153 = mcdc.universe([c1058, c1059, c1060, c1061, c1062, c1063, c1064, c1065, c1066, c1067, c1068])
u154 = mcdc.universe([c1069, c1070, c1071, c1072, c1073, c1074, c1075, c1076, c1077, c1078, c1079])
u155 = mcdc.universe([c1080, c1081])
u178 = mcdc.universe([c1104])
u2 = mcdc.universe([c2, c3, c4])
u3 = mcdc.universe([c5, c6, c7, c8])
u4 = mcdc.universe([c9, c10, c11, c12])
u6 = mcdc.universe([c16, c17, c18])
u7 = mcdc.universe([c19, c20, c21, c22])
u82 = mcdc.universe([c585, c586])
u83 = mcdc.universe([c587, c588])
u84 = mcdc.universe([c589, c590, c591, c592])
u85 = mcdc.universe([c593, c594, c595, c596, c597])
u86 = mcdc.universe([c598, c599, c600])
u87 = mcdc.universe([c601, c602, c603, c604])
u88 = mcdc.universe([c605, c606, c607, c608])

# --------------------------------------------------------------------------------------
# Cells - Level 1
# --------------------------------------------------------------------------------------

c30 = mcdc.cell(-s32, fill=u1) # Name: GT empty stack (0)
c31 = mcdc.cell(+s32 & -s33, fill=u1) # Name: GT empty stack (1)
c32 = mcdc.cell(+s33 & -s34, fill=u1) # Name: GT empty stack (2)
c33 = mcdc.cell(+s34 & -s35, fill=u6) # Name: GT empty stack (3)
c34 = mcdc.cell(+s35 & -s38, fill=u6) # Name: GT empty stack (4)
c35 = mcdc.cell(+s38 & -s39, fill=u7) # Name: GT empty stack (5)
c36 = mcdc.cell(+s39 & -s48, fill=u6) # Name: GT empty stack (6)
c37 = mcdc.cell(+s48 & -s40, fill=u2) # Name: GT empty stack (7)
c38 = mcdc.cell(+s40 & -s41, fill=u4) # Name: GT empty stack (8)
c39 = mcdc.cell(+s41 & -s42, fill=u2) # Name: GT empty stack (9)
c40 = mcdc.cell(+s42 & -s43, fill=u4) # Name: GT empty stack (10)
c41 = mcdc.cell(+s43 & -s44, fill=u2) # Name: GT empty stack (11)
c42 = mcdc.cell(+s44 & -s45, fill=u4) # Name: GT empty stack (12)
c43 = mcdc.cell(+s45 & -s36, fill=u2) # Name: GT empty stack (13)
c44 = mcdc.cell(+s36 & -s46, fill=u2) # Name: GT empty stack (14)
c45 = mcdc.cell(+s46 & -s47, fill=u4) # Name: GT empty stack (15)
c46 = mcdc.cell(+s47 & -s49, fill=u2) # Name: GT empty stack (16)
c47 = mcdc.cell(+s49 & -s50, fill=u2) # Name: GT empty stack (17)
c48 = mcdc.cell(+s50 & -s51, fill=u2) # Name: GT empty stack (18)
c49 = mcdc.cell(+s51 & -s52, fill=u1) # Name: GT empty stack (19)
c50 = mcdc.cell(+s52, fill=u1) # Name: GT empty stack (20)
c51 = mcdc.cell(-s32, fill=u1) # Name: GT empty instr (0)
c52 = mcdc.cell(+s32 & -s33, fill=u1) # Name: GT empty instr (1)
c53 = mcdc.cell(+s33 & -s34, fill=u1) # Name: GT empty instr (2)
c54 = mcdc.cell(+s34 & -s35, fill=u2) # Name: GT empty instr (3)
c55 = mcdc.cell(+s35 & -s38, fill=u2) # Name: GT empty instr (4)
c56 = mcdc.cell(+s38 & -s39, fill=u3) # Name: GT empty instr (5)
c57 = mcdc.cell(+s39 & -s48, fill=u2) # Name: GT empty instr (6)
c58 = mcdc.cell(+s48 & -s40, fill=u2) # Name: GT empty instr (7)
c59 = mcdc.cell(+s40 & -s41, fill=u4) # Name: GT empty instr (8)
c60 = mcdc.cell(+s41 & -s42, fill=u2) # Name: GT empty instr (9)
c61 = mcdc.cell(+s42 & -s43, fill=u4) # Name: GT empty instr (10)
c62 = mcdc.cell(+s43 & -s44, fill=u2) # Name: GT empty instr (11)
c63 = mcdc.cell(+s44 & -s45, fill=u4) # Name: GT empty instr (12)
c64 = mcdc.cell(+s45 & -s36, fill=u2) # Name: GT empty instr (13)
c65 = mcdc.cell(+s36 & -s46, fill=u2) # Name: GT empty instr (14)
c66 = mcdc.cell(+s46 & -s47, fill=u4) # Name: GT empty instr (15)
c67 = mcdc.cell(+s47 & -s49, fill=u2) # Name: GT empty instr (16)
c68 = mcdc.cell(+s49 & -s50, fill=u2) # Name: GT empty instr (17)
c69 = mcdc.cell(+s50 & -s51, fill=u2) # Name: GT empty instr (18)
c70 = mcdc.cell(+s51 & -s52, fill=u1) # Name: GT empty instr (19)
c71 = mcdc.cell(+s52, fill=u1) # Name: GT empty instr (20)
c613 = mcdc.cell(-s38 & +s1, fill=u86) # Name: Fuel pin (1.6%) stack (o0)
c614 = mcdc.cell(+s38 & -s39 & +s1, fill=u87) # Name: Fuel pin (1.6%) stack (o1)
c615 = mcdc.cell(+s39 & -s48 & +s1, fill=u86) # Name: Fuel pin (1.6%) stack (o2)
c616 = mcdc.cell(+s48 & -s40 & +s1, fill=u86) # Name: Fuel pin (1.6%) stack (o3)
c617 = mcdc.cell(+s40 & -s41 & +s1, fill=u88) # Name: Fuel pin (1.6%) stack (o4)
c618 = mcdc.cell(+s41 & -s42 & +s1, fill=u86) # Name: Fuel pin (1.6%) stack (o5)
c619 = mcdc.cell(+s42 & -s43 & +s1, fill=u88) # Name: Fuel pin (1.6%) stack (o6)
c620 = mcdc.cell(+s43 & -s44 & +s1, fill=u86) # Name: Fuel pin (1.6%) stack (o7)
c621 = mcdc.cell(+s44 & -s45 & +s1, fill=u88) # Name: Fuel pin (1.6%) stack (o8)
c622 = mcdc.cell(+s45 & +s1, fill=u86) # Name: Fuel pin (1.6%) stack (o9)
c624 = mcdc.cell(-s32, fill=u1) # Name: Fuel (1.6%) stack (0)
c625 = mcdc.cell(+s32 & -s33, fill=u82) # Name: Fuel (1.6%) stack (1)
c626 = mcdc.cell(+s33 & -s34, fill=u82) # Name: Fuel (1.6%) stack (2)
c627 = mcdc.cell(+s34 & -s35, fill=u83) # Name: Fuel (1.6%) stack (3)
c629 = mcdc.cell(+s36 & -s46, fill=u84) # Name: Fuel (1.6%) stack (5)
c630 = mcdc.cell(+s46 & -s47, fill=u85) # Name: Fuel (1.6%) stack (6)
c631 = mcdc.cell(+s47 & -s49, fill=u84) # Name: Fuel (1.6%) stack (7)
c632 = mcdc.cell(+s49 & -s50, fill=u83) # Name: Fuel (1.6%) stack (8)
c633 = mcdc.cell(+s50 & -s51, fill=u1) # Name: Fuel (1.6%) stack (9)
c634 = mcdc.cell(+s51 & -s52, fill=u82) # Name: Fuel (1.6%) stack (10)
c635 = mcdc.cell(+s52, fill=u1) # Name: Fuel (1.6%) stack (11)
c640 = mcdc.cell(-s38 & +s1, fill=u86) # Name: Fuel pin (2.4%) stack (o0)
c641 = mcdc.cell(+s38 & -s39 & +s1, fill=u87) # Name: Fuel pin (2.4%) stack (o1)
c642 = mcdc.cell(+s39 & -s48 & +s1, fill=u86) # Name: Fuel pin (2.4%) stack (o2)
c643 = mcdc.cell(+s48 & -s40 & +s1, fill=u86) # Name: Fuel pin (2.4%) stack (o3)
c644 = mcdc.cell(+s40 & -s41 & +s1, fill=u88) # Name: Fuel pin (2.4%) stack (o4)
c645 = mcdc.cell(+s41 & -s42 & +s1, fill=u86) # Name: Fuel pin (2.4%) stack (o5)
c646 = mcdc.cell(+s42 & -s43 & +s1, fill=u88) # Name: Fuel pin (2.4%) stack (o6)
c647 = mcdc.cell(+s43 & -s44 & +s1, fill=u86) # Name: Fuel pin (2.4%) stack (o7)
c648 = mcdc.cell(+s44 & -s45 & +s1, fill=u88) # Name: Fuel pin (2.4%) stack (o8)
c649 = mcdc.cell(+s45 & +s1, fill=u86) # Name: Fuel pin (2.4%) stack (o9)
c651 = mcdc.cell(-s32, fill=u1) # Name: Fuel (2.4%) stack (0)
c652 = mcdc.cell(+s32 & -s33, fill=u82) # Name: Fuel (2.4%) stack (1)
c653 = mcdc.cell(+s33 & -s34, fill=u82) # Name: Fuel (2.4%) stack (2)
c654 = mcdc.cell(+s34 & -s35, fill=u83) # Name: Fuel (2.4%) stack (3)
c656 = mcdc.cell(+s36 & -s46, fill=u84) # Name: Fuel (2.4%) stack (5)
c657 = mcdc.cell(+s46 & -s47, fill=u85) # Name: Fuel (2.4%) stack (6)
c658 = mcdc.cell(+s47 & -s49, fill=u84) # Name: Fuel (2.4%) stack (7)
c659 = mcdc.cell(+s49 & -s50, fill=u83) # Name: Fuel (2.4%) stack (8)
c660 = mcdc.cell(+s50 & -s51, fill=u1) # Name: Fuel (2.4%) stack (9)
c661 = mcdc.cell(+s51 & -s52, fill=u82) # Name: Fuel (2.4%) stack (10)
c662 = mcdc.cell(+s52, fill=u1) # Name: Fuel (2.4%) stack (11)
c667 = mcdc.cell(-s38 & +s1, fill=u86) # Name: Fuel pin (3.1%) stack (o0)
c668 = mcdc.cell(+s38 & -s39 & +s1, fill=u87) # Name: Fuel pin (3.1%) stack (o1)
c669 = mcdc.cell(+s39 & -s48 & +s1, fill=u86) # Name: Fuel pin (3.1%) stack (o2)
c670 = mcdc.cell(+s48 & -s40 & +s1, fill=u86) # Name: Fuel pin (3.1%) stack (o3)
c671 = mcdc.cell(+s40 & -s41 & +s1, fill=u88) # Name: Fuel pin (3.1%) stack (o4)
c672 = mcdc.cell(+s41 & -s42 & +s1, fill=u86) # Name: Fuel pin (3.1%) stack (o5)
c673 = mcdc.cell(+s42 & -s43 & +s1, fill=u88) # Name: Fuel pin (3.1%) stack (o6)
c674 = mcdc.cell(+s43 & -s44 & +s1, fill=u86) # Name: Fuel pin (3.1%) stack (o7)
c675 = mcdc.cell(+s44 & -s45 & +s1, fill=u88) # Name: Fuel pin (3.1%) stack (o8)
c676 = mcdc.cell(+s45 & +s1, fill=u86) # Name: Fuel pin (3.1%) stack (o9)
c678 = mcdc.cell(-s32, fill=u1) # Name: Fuel (3.1%) stack (0)
c679 = mcdc.cell(+s32 & -s33, fill=u82) # Name: Fuel (3.1%) stack (1)
c680 = mcdc.cell(+s33 & -s34, fill=u82) # Name: Fuel (3.1%) stack (2)
c681 = mcdc.cell(+s34 & -s35, fill=u83) # Name: Fuel (3.1%) stack (3)
c683 = mcdc.cell(+s36 & -s46, fill=u84) # Name: Fuel (3.1%) stack (5)
c684 = mcdc.cell(+s46 & -s47, fill=u85) # Name: Fuel (3.1%) stack (6)
c685 = mcdc.cell(+s47 & -s49, fill=u84) # Name: Fuel (3.1%) stack (7)
c686 = mcdc.cell(+s49 & -s50, fill=u83) # Name: Fuel (3.1%) stack (8)
c687 = mcdc.cell(+s50 & -s51, fill=u1) # Name: Fuel (3.1%) stack (9)
c688 = mcdc.cell(+s51 & -s52, fill=u82) # Name: Fuel (3.1%) stack (10)
c689 = mcdc.cell(+s52, fill=u1) # Name: Fuel (3.1%) stack (11)
c1082 = mcdc.cell(fill=u150, rotation=[-0.0, -0.0, 90.0]) # Name: reflector NE
c1083 = mcdc.cell(fill=u150, rotation=[-0.0, -0.0, -90.0]) # Name: reflector SW
c1084 = mcdc.cell(fill=u150, rotation=[-0.0, -0.0, -180.0]) # Name: reflector SE
c1085 = mcdc.cell(fill=u155, rotation=[-0.0, -180.0, 90.0]) # Name: reflector 0,2
c1086 = mcdc.cell(fill=u154, rotation=[-0.0, -0.0, 90.0]) # Name: reflector 0,3
c1087 = mcdc.cell(fill=u152, rotation=[-0.0, -0.0, 90.0]) # Name: reflector 0,4
c1088 = mcdc.cell(fill=u153, rotation=[-0.0, -0.0, 90.0]) # Name: reflector 0,5
c1089 = mcdc.cell(fill=u155, rotation=[-0.0, -0.0, 90.0]) # Name: reflector 0,6
c1090 = mcdc.cell(fill=u151, rotation=[-0.0, -0.0, 90.0]) # Name: reflector 1,7
c1091 = mcdc.cell(fill=u155, rotation=[-0.0, -180.0, -0.0]) # Name: reflector 2,8
c1092 = mcdc.cell(fill=u153, rotation=[-0.0, -180.0, -0.0]) # Name: reflector 3,8
c1093 = mcdc.cell(fill=u152, rotation=[-0.0, -180.0, -0.0]) # Name: reflector 4,8
c1094 = mcdc.cell(fill=u153, rotation=[-0.0, -0.0, -180.0]) # Name: reflector 5,8
c1095 = mcdc.cell(fill=u155, rotation=[-180.0, -0.0, -0.0]) # Name: reflector 6,0
c1096 = mcdc.cell(fill=u155, rotation=[-0.0, -0.0, -180.0]) # Name: reflector 6,8
c1097 = mcdc.cell(fill=u151, rotation=[-180.0, -0.0, -0.0]) # Name: reflector 7,1
c1098 = mcdc.cell(fill=u151, rotation=[-0.0, -0.0, -180.0]) # Name: reflector 7,7
c1099 = mcdc.cell(fill=u155, rotation=[-0.0, -0.0, -90.0]) # Name: reflector 8,2
c1100 = mcdc.cell(fill=u153, rotation=[-0.0, -0.0, -90.0]) # Name: reflector 8,3
c1101 = mcdc.cell(fill=u152, rotation=[-0.0, -0.0, -90.0]) # Name: reflector 8,4
c1102 = mcdc.cell(fill=u154, rotation=[-0.0, -0.0, -90.0]) # Name: reflector 8,5
c1103 = mcdc.cell(fill=u155, rotation=[-0.0, -0.0, -180.0]) # Name: reflector 8,6

# --------------------------------------------------------------------------------------
# Moving control rods - Level 1
# --------------------------------------------------------------------------------------

for case in cases:
    cell_cr_water = config[f'cell_cr_water_{case}']
    cell_cr_water_tube = config[f'cell_cr_water_tube_{case}']
    cell_cr_water_dashpot = config[f'cell_cr_water_dashpot_{case}']

    # Universes - Level 1
    u1_ = mcdc.universe([cell_cr_water])
    u2_ = mcdc.universe([cell_cr_water_tube, c3, c4])
    u4_ = mcdc.universe([cell_cr_water_tube, c10, c11, c12])
    u6_ = mcdc.universe([cell_cr_water_dashpot, c17, c18])
    u7_ = mcdc.universe([cell_cr_water_dashpot, c20, c21, c22])

    # Cells - Level 1
    c30_ = mcdc.cell(-s32, fill=u1_)
    c31_ = mcdc.cell(+s32 & -s33, fill=u1_)
    c32_ = mcdc.cell(+s33 & -s34, fill=u1_)
    c33_ = mcdc.cell(+s34 & -s35, fill=u6_)
    c34_ = mcdc.cell(+s35 & -s38, fill=u6_)
    c35_ = mcdc.cell(+s38 & -s39, fill=u7_)
    c36_ = mcdc.cell(+s39 & -s48, fill=u6_)
    c37_ = mcdc.cell(+s48 & -s40, fill=u2_)
    c38_ = mcdc.cell(+s40 & -s41, fill=u4_)
    c39_ = mcdc.cell(+s41 & -s42, fill=u2_)
    c40_ = mcdc.cell(+s42 & -s43, fill=u4_)
    c41_ = mcdc.cell(+s43 & -s44, fill=u2_)
    c42_ = mcdc.cell(+s44 & -s45, fill=u4_)
    c43_ = mcdc.cell(+s45 & -s36, fill=u2_)
    c44_ = mcdc.cell(+s36 & -s46, fill=u2_)
    c45_ = mcdc.cell(+s46 & -s47, fill=u4_)
    c46_ = mcdc.cell(+s47 & -s49, fill=u2_)
    c47_ = mcdc.cell(+s49 & -s50, fill=u2_)
    c48_ = mcdc.cell(+s50 & -s51, fill=u2_)
    c49_ = mcdc.cell(+s51 & -s52, fill=u1_)
    c50_ = mcdc.cell(+s52, fill=u1_)

    # Universes - Level 2
    config[f'u10_{case}'] = mcdc.universe([c30_, c31_, c32_, c33_, c34_, c35_, c36_, c37_, c38_, c39_, c40_, c41_, c42_, c43_, c44_, c45_, c46_, c47_, c48_, c49_, c50_])

    # Continue to Lattices - Level 3

# --------------------------------------------------------------------------------------
# Universes - Level 2
# --------------------------------------------------------------------------------------

u10 = mcdc.universe([c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50])
u11 = mcdc.universe([c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71])
u156 = mcdc.universe([c1082])
u157 = mcdc.universe([c1083])
u158 = mcdc.universe([c1084])
u159 = mcdc.universe([c1085])
u160 = mcdc.universe([c1086])
u161 = mcdc.universe([c1087])
u162 = mcdc.universe([c1088])
u163 = mcdc.universe([c1089])
u164 = mcdc.universe([c1090])
u165 = mcdc.universe([c1091])
u166 = mcdc.universe([c1092])
u167 = mcdc.universe([c1093])
u168 = mcdc.universe([c1094])
u169 = mcdc.universe([c1095])
u170 = mcdc.universe([c1096])
u171 = mcdc.universe([c1097])
u172 = mcdc.universe([c1098])
u173 = mcdc.universe([c1099])
u174 = mcdc.universe([c1100])
u175 = mcdc.universe([c1101])
u176 = mcdc.universe([c1102])
u177 = mcdc.universe([c1103])
u90 = mcdc.universe([c613, c614, c615, c616, c617, c618, c619, c620, c621, c622, c623])
u93 = mcdc.universe([c640, c641, c642, c643, c644, c645, c646, c647, c648, c649, c650])
u96 = mcdc.universe([c667, c668, c669, c670, c671, c672, c673, c674, c675, c676, c677])

# --------------------------------------------------------------------------------------
# Cells - Level 2
# --------------------------------------------------------------------------------------

c628 = mcdc.cell(+s35 & -s36, fill=u90) # Name: Fuel (1.6%) stack (4)
c655 = mcdc.cell(+s35 & -s36, fill=u93) # Name: Fuel (2.4%) stack (4)
c682 = mcdc.cell(+s35 & -s36, fill=u96) # Name: Fuel (3.1%) stack (4)

# --------------------------------------------------------------------------------------
# Universes - Level 3
# --------------------------------------------------------------------------------------

u91 = mcdc.universe([c624, c625, c626, c627, c628, c629, c630, c631, c632, c633, c634, c635])
u94 = mcdc.universe([c651, c652, c653, c654, c655, c656, c657, c658, c659, c660, c661, c662])
u97 = mcdc.universe([c678, c679, c680, c681, c682, c683, c684, c685, c686, c687, c688, c689])

# --------------------------------------------------------------------------------------
# Lattices - Level 3
# --------------------------------------------------------------------------------------

u10r = config['u10_r']
u10rx = config['u10_rx']
u10s = config['u10_s']
u10sx2 = config['u10_sx2']
u10sx3 = config['u10_sx3']

# Lattice name: Assembly (1.6%) no BAs
l98 = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91, u91, u91, u91],
        [u91, u91, u91, u10, u91, u91, u91, u91, u91, u91, u91, u91, u91, u10, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10, u91, u91, u10, u91, u91, u11, u91, u91, u10, u91, u91, u10, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u10, u91, u91, u91, u91, u91, u91, u91, u91, u91, u10, u91, u91, u91],
        [u91, u91, u91, u91, u91, u10, u91, u91, u10, u91, u91, u10, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
    ],
)

# Lattice name: Assembly (1.6%) no BAs, with regulating rods
l98r = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91, u91, u91, u91],
        [u91, u91, u91, u10r, u91, u91, u91, u91, u91, u91, u91, u91, u91, u10r, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10r, u91, u91, u10r, u91, u91, u11, u91, u91, u10r, u91, u91, u10r, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u10r, u91, u91, u91, u91, u91, u91, u91, u91, u91, u10r, u91, u91, u91],
        [u91, u91, u91, u91, u91, u10r, u91, u91, u10r, u91, u91, u10r, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
    ],
)

# Lattice name: Assembly (1.6%) no BAs, with stuck regulating rods
l98rx = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u91, u91, u91],
        [u91, u91, u91, u10rx, u91, u91, u91, u91, u91, u91, u91, u91, u91, u10rx, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10rx, u91, u91, u10rx, u91, u91, u11, u91, u91, u10rx, u91, u91, u10rx, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u10rx, u91, u91, u91, u91, u91, u91, u91, u91, u91, u10rx, u91, u91, u91],
        [u91, u91, u91, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u10rx, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
        [u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91, u91],
    ],
)

# Lattice name: Assembly (2.4%) no BAs
l134 = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94, u94, u94, u94],
        [u94, u94, u94, u10, u94, u94, u94, u94, u94, u94, u94, u94, u94, u10, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10, u94, u94, u10, u94, u94, u11, u94, u94, u10, u94, u94, u10, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u10, u94, u94, u94, u94, u94, u94, u94, u94, u94, u10, u94, u94, u94],
        [u94, u94, u94, u94, u94, u10, u94, u94, u10, u94, u94, u10, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
    ],
)

# Lattice name: Assembly (2.4%) no BAs, with safety rods
l134s = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94, u94, u94, u94],
        [u94, u94, u94, u10s, u94, u94, u94, u94, u94, u94, u94, u94, u94, u10s, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10s, u94, u94, u10s, u94, u94, u11, u94, u94, u10s, u94, u94, u10s, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u10s, u94, u94, u94, u94, u94, u94, u94, u94, u94, u10s, u94, u94, u94],
        [u94, u94, u94, u94, u94, u10s, u94, u94, u10s, u94, u94, u10s, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
    ],
)

# Lattice name: Assembly (2.4%) no BAs, with stuck safety rods
l134sx = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u94, u94, u94],
        [u94, u94, u94, u10sx2, u94, u94, u94, u94, u94, u94, u94, u94, u94, u10sx2, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u11, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u10sx2, u94, u94, u94, u94, u94, u94, u94, u94, u94, u10sx2, u94, u94, u94],
        [u94, u94, u94, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u10sx2, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
        [u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94, u94],
    ],
)

# Lattice name: Assembly (3.1%) no BAs
l142 = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97, u97, u97, u97],
        [u97, u97, u97, u10, u97, u97, u97, u97, u97, u97, u97, u97, u97, u10, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10, u97, u97, u10, u97, u97, u11, u97, u97, u10, u97, u97, u10, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u10, u97, u97, u97, u97, u97, u97, u97, u97, u97, u10, u97, u97, u97],
        [u97, u97, u97, u97, u97, u10, u97, u97, u10, u97, u97, u10, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
    ],
)

# Lattice name: Assembly (3.1%) no BAs, with safety rods
l142s = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97, u97, u97, u97],
        [u97, u97, u97, u10s, u97, u97, u97, u97, u97, u97, u97, u97, u97, u10s, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10s, u97, u97, u10s, u97, u97, u11, u97, u97, u10s, u97, u97, u10s, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u10s, u97, u97, u97, u97, u97, u97, u97, u97, u97, u10s, u97, u97, u97],
        [u97, u97, u97, u97, u97, u10s, u97, u97, u10s, u97, u97, u10s, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
    ],
)

# Lattice name: Assembly (3.1%) no BAs, with stuck safety rods
l142sx = mcdc.lattice(
    x = [-10.70864, 1.25984, 17],
    y = [-10.70864, 1.25984, 17],
    universes = [
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u97, u97, u97],
        [u97, u97, u97, u10sx3, u97, u97, u97, u97, u97, u97, u97, u97, u97, u10sx3, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u11, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u10sx3, u97, u97, u97, u97, u97, u97, u97, u97, u97, u10sx3, u97, u97, u97],
        [u97, u97, u97, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u10sx3, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
        [u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97, u97],
    ],
)

# --------------------------------------------------------------------------------------
# Cells - Level 3
# --------------------------------------------------------------------------------------

c690 = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l98) # Name: Assembly (1.6%) no BAs lattice
c690r = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l98r) # Name: Assembly (1.6%) no BAs lattice, with regulating rods
c690rx = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l98rx) # Name: Assembly (1.6%) no BAs lattice, with stuck regulating rods
c924 = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l134) # Name: Assembly (2.4%) no BAs lattice
c924s = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l134s) # Name: Assembly (2.4%) no BAs lattice, with safety rods
c924sx = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l134sx) # Name: Assembly (2.4%) no BAs lattice, with stuck safety rods
c976 = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l142) # Name: Assembly (3.1%) no BAs lattice
c976s = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l142s) # Name: Assembly (3.1%) no BAs lattice, with safety rods
c976sx = mcdc.cell(+s24 & -s25 & +s26 & -s27, fill=l142sx) # Name: Assembly (3.1%) no BAs lattice, with stuck safety rods

# --------------------------------------------------------------------------------------
# Universes - Level 4
# --------------------------------------------------------------------------------------

u135 = mcdc.universe([c924, c925, c926, c927, c928, c929, c930, c931, c932, c933, c934, c935, c936])
u135s = mcdc.universe([c924s, c925, c926, c927, c928, c929, c930, c931, c932, c933, c934, c935, c936])
u135sx = mcdc.universe([c924sx, c925, c926, c927, c928, c929, c930, c931, c932, c933, c934, c935, c936])
u143 = mcdc.universe([c976, c977, c978, c979, c980, c981, c982, c983, c984, c985, c986, c987, c988])
u143s = mcdc.universe([c976s, c977, c978, c979, c980, c981, c982, c983, c984, c985, c986, c987, c988])
u143sx = mcdc.universe([c976sx, c977, c978, c979, c980, c981, c982, c983, c984, c985, c986, c987, c988])
u99 = mcdc.universe([c690, c691, c692, c693, c694, c695, c696, c697, c698, c699, c700, c701, c702])
u99r = mcdc.universe([c690r, c691, c692, c693, c694, c695, c696, c697, c698, c699, c700, c701, c702])
u99rx = mcdc.universe([c690rx, c691, c692, c693, c694, c695, c696, c697, c698, c699, c700, c701, c702])

# --------------------------------------------------------------------------------------
# Lattices - Level 4
# --------------------------------------------------------------------------------------

# Lattice name: Main core
l179 = mcdc.lattice(
    x = [-96.76637999999998, 21.503639999999997, 9],
    y = [-96.76637999999998, 21.503639999999997, 9],
    universes = [
        [u178, u178,  u159,   u160,  u161,   u162,  u163,  u178,  u178],
        [u178, u151,  u150,   u143,  u143sx, u143,  u156,  u164,  u178],
        [u155, u150,  u143,   u135s, u99,    u135s, u143,  u156,  u165],
        [u153, u143,  u135s,  u99,   u99r,   u99,   u135s, u143,  u166],
        [u152, u143s, u99,    u99r,  u135,   u99rx, u99,   u143s, u167],
        [u154, u143,  u135sx, u99,   u99r,   u99,   u135s, u143,  u168],
        [u169, u157,  u143,   u135s, u99,    u135s, u143,  u158,  u170],
        [u178, u171,  u157,   u143,  u143s,  u143,  u158,  u172,  u178],
        [u178, u178,  u173,   u174,  u175,   u176,  u177,  u178,  u178],
    ],
)

# --------------------------------------------------------------------------------------
# Cells - Level 4
# --------------------------------------------------------------------------------------

c1105 = mcdc.cell(-s71 & +s81 & -s80, fill=l179) # Name: Main core

# --------------------------------------------------------------------------------------
# Universes - Level 5
# --------------------------------------------------------------------------------------

u0 = mcdc.universe([c1105, c1106, c1107, c1108], root=True)

# =============================================================================
# Set source
# =============================================================================
# In the center, 1 MeV

mcdc.source(
    z=[0.0, 200.0],
    energy=np.array([[1e6 - 1, 1e6 + 1], [1.0, 1.0]]),
    isotropic=True,
    time=[0.0, 10.0]
)

# =============================================================================
# Set tally and parameter, and then run mcdc
# =============================================================================

# Tally
xmin = -133.35
xmax = 133.35
xmin_core = -96.76637999999998
xmax_core = 96.76637999999998
Nx = 9 * 17
#
x_grid = np.linspace(xmin_core, xmax_core, Nx + 1)
pitch = x_grid[1] - x_grid[0]
x_grid_right = np.append(np.arange(xmax_core, xmax, pitch), xmax)[1:]
x_grid_left = np.flip(np.append(np.arange(xmin_core, xmin, -pitch), xmin))[:-1]
x_grid = np.concatenate((x_grid_left, x_grid, x_grid_right))
#
xf_grid = np.linspace(xmin_core, xmax_core, Nx + 1)
#
zmin = -36.6205
zmax = 246.61149999999998
zmin_core = config['bottom']
zmax_core = config['top']
Nz = math.ceil((zmax - zmin)/pitch)
Nz_core = math.ceil((zmax_core - zmin_core)/pitch)
#
z_grid = np.linspace(zmin, zmax, Nz + 1)
zf_grid = np.linspace(zmin_core, zmax_core, Nz_core + 1)
#
t_grid = np.linspace(0.0, 20.0, 201)

mcdc.tally.mesh_tally(
    scores=["fission"],
    t=t_grid,
)
'''
mcdc.tally.mesh_tally(
    scores=["fission"],
    x=xf_grid,
    y=xf_grid,
    z=zf_grid,
    t=t_grid,
)
mcdc.tally.mesh_tally(
    scores=["flux"],
    x=x_grid,
    y=x_grid,
    z=z_grid,
    E=np.array([0.0, 0.625, 2e7]),
    t=t_grid,
)
'''

# Setting
mcdc.setting(N_particle=100000, active_bank_buff=10000, time_boundary=10.0)

# Run
mcdc.run()

'''
colors = {
    m_helium: "azure",
    m_inconel: "gray",
    m_ss302: "sienna",
    m_ss304: "tan",
    m_csteel: "olive",
    m_zr4: "slategray",
    m_m5: "black",
    m_fuel31: "red",
    m_fuel24: "orange",
    m_fuel16: "gold",
    m_water: 'blue',
    m_cr: 'green',
}
#mcdc.visualize('xy', x=[-150, 150], y=[-150, 150], z=100.0, pixels=(1000, 1000), colors=colors)
#mcdc.visualize('xz', x=[-150, 150], z=[-37, 247], y=0.0, pixels=(1000, 1000), colors=colors)
'''
