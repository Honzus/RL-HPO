import re

log_text = """[I 2025-10-20 19:12:02,162] Trial 0 finished with value: 0.0 and parameters: {'epsilon': 0.5488135039273248, 'learning_rate': 0.7151893663724195}. Best is trial 0 with value: 0.0.
[I 2025-10-20 19:12:05,342] Trial 1 finished with value: 1.0414999999999999 and parameters: {'epsilon': 0.6027633760716439, 'learning_rate': 0.5448831829968969}. Best is trial 1 with value: 1.0414999999999999.
[I 2025-10-20 19:12:08,198] Trial 2 finished with value: 0.506 and parameters: {'epsilon': 0.4236547993389047, 'learning_rate': 0.6458941130666561}. Best is trial 1 with value: 1.0414999999999999.
[I 2025-10-20 19:12:11,285] Trial 3 finished with value: 0.3545 and parameters: {'epsilon': 0.4375872112626925, 'learning_rate': 0.8917730007820798}. Best is trial 1 with value: 1.0414999999999999.
[I 2025-10-20 19:12:14,622] Trial 4 finished with value: 1.4324999999999999 and parameters: {'epsilon': 0.9636627605010293, 'learning_rate': 0.3834415188257777}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:17,846] Trial 5 finished with value: 1.275 and parameters: {'epsilon': 0.7917250380826646, 'learning_rate': 0.5288949197529045}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:21,373] Trial 6 finished with value: 0.0 and parameters: {'epsilon': 0.5680445610939323, 'learning_rate': 0.925596638292661}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:23,666] Trial 7 finished with value: 0.0 and parameters: {'epsilon': 0.07103605819788694, 'learning_rate': 0.08712929970154071}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:25,893] Trial 8 finished with value: 0.0 and parameters: {'epsilon': 0.02021839744032572, 'learning_rate': 0.832619845547938}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:29,460] Trial 9 finished with value: 0.0 and parameters: {'epsilon': 0.7781567509498505, 'learning_rate': 0.8700121482468192}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:33,668] Trial 10 finished with value: 1.2439999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.43285905046925377}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:36,329] Trial 11 finished with value: 0.0 and parameters: {'epsilon': 0.10786609554606869, 'learning_rate': 0.4142374851568551}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:40,095] Trial 12 finished with value: 1.0465 and parameters: {'epsilon': 0.7609548505776217, 'learning_rate': 0.34559846197783706}. Best is trial 4 with value: 1.4324999999999999.
[I 2025-10-20 19:12:43,753] Trial 13 finished with value: 1.6755 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.08893484590722764}. Best is trial 13 with value: 1.6755.
[I 2025-10-20 19:12:47,083] Trial 14 finished with value: 0.0 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.0}. Best is trial 13 with value: 1.6755.
[I 2025-10-20 19:12:50,581] Trial 15 finished with value: 1.5675000000000001 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17373874367477693}. Best is trial 13 with value: 1.6755.
[I 2025-10-20 19:12:54,026] Trial 16 finished with value: 1.35 and parameters: {'epsilon': 1.0, 'learning_rate': 0.134356177396521}. Best is trial 13 with value: 1.6755.
[I 2025-10-20 19:12:57,611] Trial 17 finished with value: 1.3635 and parameters: {'epsilon': 1.0, 'learning_rate': 0.22758192617839546}. Best is trial 13 with value: 1.6755.
[I 2025-10-20 19:13:00,922] Trial 18 finished with value: 0.892 and parameters: {'epsilon': 1.0, 'learning_rate': 0.5510373548969177}. Best is trial 13 with value: 1.6755.
[I 2025-10-20 19:13:04,480] Trial 19 finished with value: 1.7015 and parameters: {'epsilon': 1.0, 'learning_rate': 0.16951621927952137}. Best is trial 19 with value: 1.7015.
[I 2025-10-20 19:13:08,164] Trial 20 finished with value: 1.5910000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.16741454400244352}. Best is trial 19 with value: 1.7015.
[I 2025-10-20 19:13:11,825] Trial 21 finished with value: 1.7805 and parameters: {'epsilon': 0.8036560832038605, 'learning_rate': 0.16132227707263835}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:15,064] Trial 22 finished with value: 1.7105000000000001 and parameters: {'epsilon': 0.7658432948195547, 'learning_rate': 0.15997802007755782}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:18,356] Trial 23 finished with value: 1.405 and parameters: {'epsilon': 0.7880380796007673, 'learning_rate': 0.14775324389037237}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:21,893] Trial 24 finished with value: 1.5835 and parameters: {'epsilon': 0.7396664175089229, 'learning_rate': 0.2023639216638168}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:24,183] Trial 25 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 1.0}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:27,660] Trial 26 finished with value: 1.7195 and parameters: {'epsilon': 0.8224544230652993, 'learning_rate': 0.19152310503238645}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:31,117] Trial 27 finished with value: 1.6075 and parameters: {'epsilon': 0.8377978759767567, 'learning_rate': 0.1822787399202055}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:34,416] Trial 28 finished with value: 1.4245 and parameters: {'epsilon': 1.0, 'learning_rate': 0.2708902254647482}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:38,374] Trial 29 finished with value: 1.353 and parameters: {'epsilon': 0.6338202855999611, 'learning_rate': 0.20362297266703422}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:42,082] Trial 30 finished with value: 1.4655 and parameters: {'epsilon': 1.0, 'learning_rate': 0.16467077395806984}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:45,972] Trial 31 finished with value: 1.751 and parameters: {'epsilon': 0.7956039826611515, 'learning_rate': 0.13662705403782094}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:49,276] Trial 32 finished with value: 1.6199999999999999 and parameters: {'epsilon': 0.7977580613365496, 'learning_rate': 0.13557519172905053}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:52,821] Trial 33 finished with value: 1.7445 and parameters: {'epsilon': 0.8211607267004726, 'learning_rate': 0.13946893478678135}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:13:56,299] Trial 34 finished with value: 1.5835000000000001 and parameters: {'epsilon': 0.7732121607375579, 'learning_rate': 0.12975157393315925}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:00,185] Trial 35 finished with value: 1.5810000000000002 and parameters: {'epsilon': 0.8599736332813518, 'learning_rate': 0.15850510244677063}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:03,439] Trial 36 finished with value: 0.0 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 1.0}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:05,795] Trial 37 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 0.5885892231846303}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:09,465] Trial 38 finished with value: 1.3565 and parameters: {'epsilon': 0.8570662354631288, 'learning_rate': 0.16340662900107095}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:12,850] Trial 39 finished with value: 1.5550000000000002 and parameters: {'epsilon': 0.6790721550107629, 'learning_rate': 0.12706490054513914}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:16,291] Trial 40 finished with value: 1.5615 and parameters: {'epsilon': 0.6358721537403522, 'learning_rate': 0.15541931223624095}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:19,602] Trial 41 finished with value: 1.6830000000000003 and parameters: {'epsilon': 0.8202606893877826, 'learning_rate': 0.1225159442910729}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:23,006] Trial 42 finished with value: 1.7395 and parameters: {'epsilon': 0.7514337618461164, 'learning_rate': 0.11179222654111128}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:26,485] Trial 43 finished with value: 1.7195 and parameters: {'epsilon': 0.7346312522425301, 'learning_rate': 0.10866478425266038}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:29,723] Trial 44 finished with value: 1.7149999999999999 and parameters: {'epsilon': 0.8346119640612188, 'learning_rate': 0.11960811903375874}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:33,277] Trial 45 finished with value: 1.7289999999999999 and parameters: {'epsilon': 0.7181267188256382, 'learning_rate': 0.10302969271833419}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:37,005] Trial 46 finished with value: 1.7205 and parameters: {'epsilon': 0.8187765024542086, 'learning_rate': 0.11264014010744629}. Best is trial 21 with value: 1.7805.
[I 2025-10-20 19:14:40,549] Trial 47 finished with value: 1.7954999999999999 and parameters: {'epsilon': 0.715204600875432, 'learning_rate': 0.09823890488532695}. Best is trial 47 with value: 1.7954999999999999.
[I 2025-10-20 19:14:44,408] Trial 48 finished with value: 1.691 and parameters: {'epsilon': 0.6780476795599527, 'learning_rate': 0.08901385597026966}. Best is trial 47 with value: 1.7954999999999999.
[I 2025-10-20 19:14:47,556] Trial 49 finished with value: 1.7429999999999999 and parameters: {'epsilon': 0.7879895963584198, 'learning_rate': 0.09839048233490837}. Best is trial 47 with value: 1.7954999999999999.
Run 1 completed:
  Best value: 1.7955
  Best params: {'epsilon': 0.715204600875432, 'learning_rate': 0.09823890488532695}

=== Starting Run 2/5 ===
/Users/janrichtr/Desktop/Thesis/Thesis/bo.py:100: ExperimentalWarning: GPSampler is experimental (supported from v3.6.0). The interface can change in the future.
  study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.GPSampler(seed=run_seed))
[I 2025-10-20 19:14:47,556] A new study created in memory with name: no-name-5e3ddef8-083e-44bc-aa4b-e4e557aeb36b
[I 2025-10-20 19:14:51,339] Trial 0 finished with value: 0.175 and parameters: {'epsilon': 0.417022004702574, 'learning_rate': 0.7203244934421581}. Best is trial 0 with value: 0.175.
[I 2025-10-20 19:14:53,505] Trial 1 finished with value: 0.0 and parameters: {'epsilon': 0.00011437481734488664, 'learning_rate': 0.30233257263183977}. Best is trial 0 with value: 0.175.
[I 2025-10-20 19:14:56,377] Trial 2 finished with value: 0.6625 and parameters: {'epsilon': 0.14675589081711304, 'learning_rate': 0.0923385947687978}. Best is trial 2 with value: 0.6625.
[I 2025-10-20 19:14:59,387] Trial 3 finished with value: 0.6955 and parameters: {'epsilon': 0.1862602113776709, 'learning_rate': 0.34556072704304774}. Best is trial 3 with value: 0.6955.
[I 2025-10-20 19:15:02,478] Trial 4 finished with value: 0.36550000000000005 and parameters: {'epsilon': 0.39676747423066994, 'learning_rate': 0.538816734003357}. Best is trial 3 with value: 0.6955.
[I 2025-10-20 19:15:05,702] Trial 5 finished with value: 0.02 and parameters: {'epsilon': 0.4191945144032948, 'learning_rate': 0.6852195003967595}. Best is trial 3 with value: 0.6955.
[I 2025-10-20 19:15:09,412] Trial 6 finished with value: 0.344 and parameters: {'epsilon': 0.20445224973151743, 'learning_rate': 0.8781174363909454}. Best is trial 3 with value: 0.6955.
[I 2025-10-20 19:15:11,624] Trial 7 finished with value: 0.0 and parameters: {'epsilon': 0.027387593197926163, 'learning_rate': 0.6704675101784022}. Best is trial 3 with value: 0.6955.
[I 2025-10-20 19:15:15,105] Trial 8 finished with value: 0.7190000000000001 and parameters: {'epsilon': 0.41730480236712697, 'learning_rate': 0.5586898284457517}. Best is trial 8 with value: 0.7190000000000001.
[I 2025-10-20 19:15:18,834] Trial 9 finished with value: 0.8655000000000002 and parameters: {'epsilon': 0.14038693859523377, 'learning_rate': 0.1981014890848788}. Best is trial 9 with value: 0.8655000000000002.
[I 2025-10-20 19:15:22,521] Trial 10 finished with value: 0.3645 and parameters: {'epsilon': 0.3519334348634639, 'learning_rate': 0.15681777113477993}. Best is trial 9 with value: 0.8655000000000002.
[I 2025-10-20 19:15:25,812] Trial 11 finished with value: 1.3615 and parameters: {'epsilon': 0.8459587788005799, 'learning_rate': 0.33330309066437874}. Best is trial 11 with value: 1.3615.
[I 2025-10-20 19:15:29,543] Trial 12 finished with value: 1.206 and parameters: {'epsilon': 0.9648868808029323, 'learning_rate': 0.25584898583051735}. Best is trial 11 with value: 1.3615.
[I 2025-10-20 19:15:32,871] Trial 13 finished with value: 0.0 and parameters: {'epsilon': 0.8249426151638204, 'learning_rate': 0.0}. Best is trial 11 with value: 1.3615.
[I 2025-10-20 19:15:36,250] Trial 14 finished with value: 0.8210000000000001 and parameters: {'epsilon': 0.9488537971022083, 'learning_rate': 0.44105509638620394}. Best is trial 11 with value: 1.3615.
[I 2025-10-20 19:15:40,374] Trial 15 finished with value: 1.383 and parameters: {'epsilon': 0.7329967204835589, 'learning_rate': 0.3512036163002038}. Best is trial 15 with value: 1.383.
[I 2025-10-20 19:15:44,046] Trial 16 finished with value: 0.687 and parameters: {'epsilon': 0.7783378793295922, 'learning_rate': 0.3109231155206678}. Best is trial 15 with value: 1.383.
[I 2025-10-20 19:15:47,406] Trial 17 finished with value: 0.0 and parameters: {'epsilon': 1.0, 'learning_rate': 1.0}. Best is trial 15 with value: 1.383.
[I 2025-10-20 19:15:51,084] Trial 18 finished with value: 1.3239999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.33517750353086756}. Best is trial 15 with value: 1.383.
[I 2025-10-20 19:15:54,310] Trial 19 finished with value: 1.4025 and parameters: {'epsilon': 1.0, 'learning_rate': 0.3383080079013623}. Best is trial 19 with value: 1.4025.
[I 2025-10-20 19:15:57,570] Trial 20 finished with value: 1.2194999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.3402466424005026}. Best is trial 19 with value: 1.4025.
[I 2025-10-20 19:16:00,924] Trial 21 finished with value: 1.568 and parameters: {'epsilon': 1.0, 'learning_rate': 0.3262773904951361}. Best is trial 21 with value: 1.568.
[I 2025-10-20 19:16:03,251] Trial 22 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 1.0}. Best is trial 21 with value: 1.568.
[I 2025-10-20 19:16:06,993] Trial 23 finished with value: 1.6779999999999997 and parameters: {'epsilon': 1.0, 'learning_rate': 0.4368668407944492}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:10,528] Trial 24 finished with value: 0.869 and parameters: {'epsilon': 1.0, 'learning_rate': 0.512162723302046}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:14,049] Trial 25 finished with value: 1.3105 and parameters: {'epsilon': 1.0, 'learning_rate': 0.23877417540176876}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:16,346] Trial 26 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 0.0}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:19,584] Trial 27 finished with value: 1.142 and parameters: {'epsilon': 1.0, 'learning_rate': 0.38758224011136927}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:23,390] Trial 28 finished with value: 1.2049999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.23094938291494113}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:26,967] Trial 29 finished with value: 1.5305 and parameters: {'epsilon': 1.0, 'learning_rate': 0.23394964264600254}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:30,585] Trial 30 finished with value: 1.2335 and parameters: {'epsilon': 1.0, 'learning_rate': 0.23597189653815315}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:34,026] Trial 31 finished with value: 0.8560000000000001 and parameters: {'epsilon': 1.0, 'learning_rate': 0.36239299766914135}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:37,463] Trial 32 finished with value: 0.5235 and parameters: {'epsilon': 1.0, 'learning_rate': 0.6509604821081195}. Best is trial 23 with value: 1.6779999999999997.
[I 2025-10-20 19:16:40,886] Trial 33 finished with value: 1.7035 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1894418755170592}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:16:44,496] Trial 34 finished with value: 1.6875 and parameters: {'epsilon': 1.0, 'learning_rate': 0.18028572455753605}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:16:48,036] Trial 35 finished with value: 1.5925 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17953548508255954}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:16:51,642] Trial 36 finished with value: 1.483 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17772183943147776}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:16:55,158] Trial 37 finished with value: 1.5314999999999999 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17405843679482796}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:16:58,924] Trial 38 finished with value: 1.6425 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.16631488983063808}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:17:03,376] Trial 39 finished with value: 1.6695 and parameters: {'epsilon': 1.0, 'learning_rate': 0.15731947777470656}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:17:07,246] Trial 40 finished with value: 1.6809999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1519683811833376}. Best is trial 33 with value: 1.7035.
[I 2025-10-20 19:17:10,879] Trial 41 finished with value: 1.7100000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14911880891354837}. Best is trial 41 with value: 1.7100000000000002.
[I 2025-10-20 19:17:14,458] Trial 42 finished with value: 1.69 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14624856750959314}. Best is trial 41 with value: 1.7100000000000002.
[I 2025-10-20 19:17:17,767] Trial 43 finished with value: 1.6595 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14539671968619314}. Best is trial 41 with value: 1.7100000000000002.
[I 2025-10-20 19:17:21,046] Trial 44 finished with value: 1.6674999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14503077585693464}. Best is trial 41 with value: 1.7100000000000002.
[I 2025-10-20 19:17:24,777] Trial 45 finished with value: 1.6355 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14475819256309125}. Best is trial 41 with value: 1.7100000000000002.
[I 2025-10-20 19:17:28,385] Trial 46 finished with value: 1.702 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14475579698469873}. Best is trial 41 with value: 1.7100000000000002.
[I 2025-10-20 19:17:32,018] Trial 47 finished with value: 1.722 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14441343511095653}. Best is trial 47 with value: 1.722.
[I 2025-10-20 19:17:35,351] Trial 48 finished with value: 1.5495 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1411807445728275}. Best is trial 47 with value: 1.722.
[I 2025-10-20 19:17:38,935] Trial 49 finished with value: 1.6490000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14198432321887}. Best is trial 47 with value: 1.722.
Run 2 completed:
  Best value: 1.7220
  Best params: {'epsilon': 1.0, 'learning_rate': 0.14441343511095653}

=== Starting Run 3/5 ===
[I 2025-10-20 19:17:38,936] A new study created in memory with name: no-name-d8eaf40f-7268-48fb-94a0-1c280c5aba92
[I 2025-10-20 19:17:42,143] Trial 0 finished with value: 1.6744999999999997 and parameters: {'epsilon': 0.43599490214200376, 'learning_rate': 0.025926231827891333}. Best is trial 0 with value: 1.6744999999999997.
[I 2025-10-20 19:17:45,869] Trial 1 finished with value: 0.6789999999999999 and parameters: {'epsilon': 0.5496624778787091, 'learning_rate': 0.4353223926182769}. Best is trial 0 with value: 1.6744999999999997.
[I 2025-10-20 19:17:48,841] Trial 2 finished with value: 1.2095 and parameters: {'epsilon': 0.42036780208748903, 'learning_rate': 0.3303348210038741}. Best is trial 0 with value: 1.6744999999999997.
[I 2025-10-20 19:17:51,915] Trial 3 finished with value: 1.026 and parameters: {'epsilon': 0.2046486340378425, 'learning_rate': 0.6192709663506637}. Best is trial 0 with value: 1.6744999999999997.
[I 2025-10-20 19:17:55,186] Trial 4 finished with value: 1.6844999999999999 and parameters: {'epsilon': 0.29965467367452314, 'learning_rate': 0.26682727510286663}. Best is trial 4 with value: 1.6844999999999999.
[I 2025-10-20 19:17:58,509] Trial 5 finished with value: 0.18 and parameters: {'epsilon': 0.6211338327692949, 'learning_rate': 0.5291420942770391}. Best is trial 4 with value: 1.6844999999999999.
[I 2025-10-20 19:18:01,526] Trial 6 finished with value: 0.19 and parameters: {'epsilon': 0.13457994534493356, 'learning_rate': 0.5135781212657464}. Best is trial 4 with value: 1.6844999999999999.
[I 2025-10-20 19:18:05,104] Trial 7 finished with value: 0.7195 and parameters: {'epsilon': 0.18443986564691528, 'learning_rate': 0.7853351478166735}. Best is trial 4 with value: 1.6844999999999999.
[I 2025-10-20 19:18:08,923] Trial 8 finished with value: 0.5475000000000001 and parameters: {'epsilon': 0.8539752926394888, 'learning_rate': 0.4942368373819278}. Best is trial 4 with value: 1.6844999999999999.
[I 2025-10-20 19:18:12,444] Trial 9 finished with value: 1.7559999999999998 and parameters: {'epsilon': 0.846561485357468, 'learning_rate': 0.079645477009061}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:15,804] Trial 10 finished with value: 0.8870000000000001 and parameters: {'epsilon': 0.2952896592560006, 'learning_rate': 0.15966884611419194}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:19,142] Trial 11 finished with value: 0.0 and parameters: {'epsilon': 1.0, 'learning_rate': 0.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:22,760] Trial 12 finished with value: 1.2315 and parameters: {'epsilon': 0.6938083348693502, 'learning_rate': 0.09758760323653255}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:26,381] Trial 13 finished with value: 1.3820000000000001 and parameters: {'epsilon': 0.30336917282415565, 'learning_rate': 0.39785860834358033}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:30,312] Trial 14 finished with value: 1.5619999999999998 and parameters: {'epsilon': 0.8394186631239945, 'learning_rate': 0.17053557612280312}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:33,757] Trial 15 finished with value: 0.0 and parameters: {'epsilon': 0.5161815691037261, 'learning_rate': 0.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:37,485] Trial 16 finished with value: 0.0 and parameters: {'epsilon': 0.7954688643170781, 'learning_rate': 0.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:41,063] Trial 17 finished with value: 1.7175 and parameters: {'epsilon': 0.34331935945675296, 'learning_rate': 0.013953679552760093}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:44,347] Trial 18 finished with value: 1.718 and parameters: {'epsilon': 0.40014289155010246, 'learning_rate': 0.06053948247397577}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:47,755] Trial 19 finished with value: 1.7064999999999997 and parameters: {'epsilon': 0.8292522965348075, 'learning_rate': 0.11676810589036121}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:51,690] Trial 20 finished with value: 0.1705 and parameters: {'epsilon': 1.0, 'learning_rate': 0.9999999999999999}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:54,057] Trial 21 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 0.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:18:57,534] Trial 22 finished with value: 0.0 and parameters: {'epsilon': 0.47613189339382794, 'learning_rate': 1.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:01,037] Trial 23 finished with value: 1.3685 and parameters: {'epsilon': 0.627743532414277, 'learning_rate': 0.21123769400858178}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:04,357] Trial 24 finished with value: 0.686 and parameters: {'epsilon': 0.2231799598239791, 'learning_rate': 0.31068704719793805}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:06,731] Trial 25 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 1.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:09,973] Trial 26 finished with value: 1.45 and parameters: {'epsilon': 0.7632400439554405, 'learning_rate': 0.16905408706292277}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:13,535] Trial 27 finished with value: 1.3545 and parameters: {'epsilon': 0.4397603660736096, 'learning_rate': 0.17285908719728774}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:16,705] Trial 28 finished with value: 0.0 and parameters: {'epsilon': 0.3721564114011392, 'learning_rate': 0.0}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:19,091] Trial 29 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 0.10930414883890244}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:22,641] Trial 30 finished with value: 1.7495 and parameters: {'epsilon': 1.0, 'learning_rate': 0.24107382476427155}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:26,794] Trial 31 finished with value: 1.6664999999999999 and parameters: {'epsilon': 1.0, 'learning_rate': 0.23688346581671474}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:30,675] Trial 32 finished with value: 1.5429999999999997 and parameters: {'epsilon': 1.0, 'learning_rate': 0.22994402900875358}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:34,311] Trial 33 finished with value: 1.375 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.20919485526174478}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:38,370] Trial 34 finished with value: 1.3815000000000002 and parameters: {'epsilon': 0.7687405672398694, 'learning_rate': 0.24591239347981567}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:42,045] Trial 35 finished with value: 1.026 and parameters: {'epsilon': 1.0, 'learning_rate': 0.3030468615109695}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:45,619] Trial 36 finished with value: 1.5470000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17089598923149038}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:49,530] Trial 37 finished with value: 1.7155 and parameters: {'epsilon': 0.7290002302263692, 'learning_rate': 0.1718633562744452}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:53,023] Trial 38 finished with value: 1.5724999999999998 and parameters: {'epsilon': 0.7298902659973266, 'learning_rate': 0.16919239808963174}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:19:56,765] Trial 39 finished with value: 0.0 and parameters: {'epsilon': 1.0, 'learning_rate': 0.7433592481016821}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:00,194] Trial 40 finished with value: 1.3495000000000001 and parameters: {'epsilon': 0.7089877772211849, 'learning_rate': 0.18845542836917295}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:03,625] Trial 41 finished with value: 1.5045000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1738734248789858}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:07,043] Trial 42 finished with value: 1.698 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17345734181109276}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:10,689] Trial 43 finished with value: 1.5130000000000001 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1682444080817241}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:14,991] Trial 44 finished with value: 1.5159999999999998 and parameters: {'epsilon': 0.6754619430611682, 'learning_rate': 0.1482462576031376}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:18,727] Trial 45 finished with value: 1.7185000000000001 and parameters: {'epsilon': 1.0, 'learning_rate': 0.16352947595654424}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:22,363] Trial 46 finished with value: 1.7109999999999999 and parameters: {'epsilon': 1.0, 'learning_rate': 0.15546493236963083}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:25,810] Trial 47 finished with value: 1.6150000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14792906878968706}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:29,280] Trial 48 finished with value: 1.5314999999999999 and parameters: {'epsilon': 0.6561748586651652, 'learning_rate': 0.12211265218666709}. Best is trial 9 with value: 1.7559999999999998.
[I 2025-10-20 19:20:33,522] Trial 49 finished with value: 1.7195 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14104741287560005}. Best is trial 9 with value: 1.7559999999999998.
Run 3 completed:
  Best value: 1.7560
  Best params: {'epsilon': 0.846561485357468, 'learning_rate': 0.079645477009061}

=== Starting Run 4/5 ===
[I 2025-10-20 19:20:33,523] A new study created in memory with name: no-name-bddad62e-b6d1-468a-91ba-05a0a9018334
[I 2025-10-20 19:20:37,223] Trial 0 finished with value: 0.5235000000000001 and parameters: {'epsilon': 0.5507979025745755, 'learning_rate': 0.7081478226181048}. Best is trial 0 with value: 0.5235000000000001.
[I 2025-10-20 19:20:40,312] Trial 1 finished with value: 0.020999999999999998 and parameters: {'epsilon': 0.2909047389129443, 'learning_rate': 0.510827605197663}. Best is trial 0 with value: 0.5235000000000001.
[I 2025-10-20 19:20:43,635] Trial 2 finished with value: 0.0 and parameters: {'epsilon': 0.8929469543476547, 'learning_rate': 0.8962930889334381}. Best is trial 0 with value: 0.5235000000000001.
[I 2025-10-20 19:20:45,970] Trial 3 finished with value: 0.0 and parameters: {'epsilon': 0.12558531046383625, 'learning_rate': 0.20724287813818676}. Best is trial 0 with value: 0.5235000000000001.
[I 2025-10-20 19:20:48,224] Trial 4 finished with value: 0.0 and parameters: {'epsilon': 0.05146720330082988, 'learning_rate': 0.44080984365063647}. Best is trial 0 with value: 0.5235000000000001.
[I 2025-10-20 19:20:50,454] Trial 5 finished with value: 0.0 and parameters: {'epsilon': 0.029876210878566956, 'learning_rate': 0.4568332243947111}. Best is trial 0 with value: 0.5235000000000001.
[I 2025-10-20 19:20:54,337] Trial 6 finished with value: 1.214 and parameters: {'epsilon': 0.6491440476147607, 'learning_rate': 0.2784872826479753}. Best is trial 6 with value: 1.214.
[I 2025-10-20 19:20:57,681] Trial 7 finished with value: 0.688 and parameters: {'epsilon': 0.6762549019801313, 'learning_rate': 0.5908628174163508}. Best is trial 6 with value: 1.214.
[I 2025-10-20 19:20:59,904] Trial 8 finished with value: 0.0 and parameters: {'epsilon': 0.023981882377165364, 'learning_rate': 0.558854087990882}. Best is trial 6 with value: 1.214.
[I 2025-10-20 19:21:03,377] Trial 9 finished with value: 0.8480000000000001 and parameters: {'epsilon': 0.2592524469074654, 'learning_rate': 0.41510119701006964}. Best is trial 6 with value: 1.214.
[I 2025-10-20 19:21:06,864] Trial 10 finished with value: 1.7200000000000002 and parameters: {'epsilon': 0.7782106639752006, 'learning_rate': 0.04315946116821251}. Best is trial 10 with value: 1.7200000000000002.
[I 2025-10-20 19:21:10,980] Trial 11 finished with value: 0.0 and parameters: {'epsilon': 1.0, 'learning_rate': 0.0}. Best is trial 10 with value: 1.7200000000000002.
[I 2025-10-20 19:21:14,163] Trial 12 finished with value: 0.0 and parameters: {'epsilon': 0.69055800133529, 'learning_rate': 0.0}. Best is trial 10 with value: 1.7200000000000002.
[I 2025-10-20 19:21:18,233] Trial 13 finished with value: 1.54 and parameters: {'epsilon': 0.8104961249383155, 'learning_rate': 0.14465812918403997}. Best is trial 10 with value: 1.7200000000000002.
[I 2025-10-20 19:21:21,999] Trial 14 finished with value: 1.721 and parameters: {'epsilon': 0.8190220557090451, 'learning_rate': 0.022208180610140727}. Best is trial 14 with value: 1.721.
[I 2025-10-20 19:21:25,383] Trial 15 finished with value: 1.7224999999999997 and parameters: {'epsilon': 0.7996580046503334, 'learning_rate': 0.058948515197743905}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:21:28,601] Trial 16 finished with value: 1.7090000000000003 and parameters: {'epsilon': 0.33916598156009925, 'learning_rate': 0.15517572275176283}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:21:31,346] Trial 17 finished with value: 0.0 and parameters: {'epsilon': 0.3179263990566222, 'learning_rate': 0.0}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:21:35,003] Trial 18 finished with value: 1.5015 and parameters: {'epsilon': 0.36107541009841665, 'learning_rate': 0.24831547988542904}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:21:38,642] Trial 19 finished with value: 1.3565 and parameters: {'epsilon': 0.30335844023146485, 'learning_rate': 0.21074192469442207}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:21:42,015] Trial 20 finished with value: 1.3945 and parameters: {'epsilon': 0.39817610165368106, 'learning_rate': 0.1695651606983407}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:26:41,065] Trial 21 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 1.0}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:26:44,410] Trial 22 finished with value: 1.0605 and parameters: {'epsilon': 0.7925622812513073, 'learning_rate': 0.36614151424086794}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:38:48,946] Trial 23 finished with value: 1.6954999999999998 and parameters: {'epsilon': 0.8115213382634704, 'learning_rate': 0.05842662336441061}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:38:54,194] Trial 24 finished with value: 0.0 and parameters: {'epsilon': 0.47713857240484076, 'learning_rate': 1.0}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:38:58,082] Trial 25 finished with value: 1.6989999999999998 and parameters: {'epsilon': 0.798898618666948, 'learning_rate': 0.006536558783287773}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:39:01,841] Trial 26 finished with value: 1.0074999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.4929735504454884}. Best is trial 15 with value: 1.7224999999999997.
[I 2025-10-20 19:39:05,223] Trial 27 finished with value: 1.725 and parameters: {'epsilon': 0.745319687401931, 'learning_rate': 0.19379169340135494}. Best is trial 27 with value: 1.725.
[I 2025-10-20 19:39:08,867] Trial 28 finished with value: 1.7335 and parameters: {'epsilon': 0.7607883322555961, 'learning_rate': 0.134591304941796}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:12,160] Trial 29 finished with value: 1.5665 and parameters: {'epsilon': 0.3512216895773962, 'learning_rate': 0.17757223862208577}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:15,428] Trial 30 finished with value: 1.2254999999999998 and parameters: {'epsilon': 0.43694141700494965, 'learning_rate': 0.34894407164056374}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:19,127] Trial 31 finished with value: 1.535 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.31775375568305325}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:22,414] Trial 32 finished with value: 1.6969999999999998 and parameters: {'epsilon': 0.9051483802801635, 'learning_rate': 0.29017139098694}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:26,508] Trial 33 finished with value: 1.0550000000000002 and parameters: {'epsilon': 0.9306185543156392, 'learning_rate': 0.3511227419521338}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:29,922] Trial 34 finished with value: 0.9915 and parameters: {'epsilon': 0.9607710111881492, 'learning_rate': 0.2439390258558496}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:33,626] Trial 35 finished with value: 1.3954999999999997 and parameters: {'epsilon': 0.8272645636359858, 'learning_rate': 0.2656820628904653}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:37,773] Trial 36 finished with value: 0.5005 and parameters: {'epsilon': 1.0, 'learning_rate': 0.6577695605153211}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:41,308] Trial 37 finished with value: 1.546 and parameters: {'epsilon': 0.7069115380912943, 'learning_rate': 0.1718986050606237}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:43,699] Trial 38 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 0.0}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:47,161] Trial 39 finished with value: 1.3865 and parameters: {'epsilon': 0.7592686436659548, 'learning_rate': 0.1667747191547672}. Best is trial 28 with value: 1.7335.
[I 2025-10-20 19:39:50,611] Trial 40 finished with value: 1.7384999999999997 and parameters: {'epsilon': 0.816574440835648, 'learning_rate': 0.08526271561504743}. Best is trial 40 with value: 1.7384999999999997.
[I 2025-10-20 19:39:54,461] Trial 41 finished with value: 1.7154999999999998 and parameters: {'epsilon': 0.815521092357418, 'learning_rate': 0.08181018929775448}. Best is trial 40 with value: 1.7384999999999997.
[I 2025-10-20 19:39:58,313] Trial 42 finished with value: 1.7469999999999999 and parameters: {'epsilon': 0.8143534683672761, 'learning_rate': 0.07596940384197835}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:02,041] Trial 43 finished with value: 1.705 and parameters: {'epsilon': 0.8166978984046148, 'learning_rate': 0.06606479422805213}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:05,587] Trial 44 finished with value: 1.7080000000000002 and parameters: {'epsilon': 0.8275778267533472, 'learning_rate': 0.02646220968246904}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:09,470] Trial 45 finished with value: 1.7120000000000002 and parameters: {'epsilon': 0.8076189155633978, 'learning_rate': 0.06859206663406471}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:12,707] Trial 46 finished with value: 1.6829999999999998 and parameters: {'epsilon': 0.8090707589359631, 'learning_rate': 0.05908135071458417}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:16,437] Trial 47 finished with value: 0.0 and parameters: {'epsilon': 0.8291129917346294, 'learning_rate': 0.0}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:20,592] Trial 48 finished with value: 1.7195 and parameters: {'epsilon': 0.8139107335282211, 'learning_rate': 0.13716709824923243}. Best is trial 42 with value: 1.7469999999999999.
[I 2025-10-20 19:40:24,397] Trial 49 finished with value: 1.5604999999999998 and parameters: {'epsilon': 0.812599023305946, 'learning_rate': 0.13557839471266642}. Best is trial 42 with value: 1.7469999999999999.
Run 4 completed:
  Best value: 1.7470
  Best params: {'epsilon': 0.8143534683672761, 'learning_rate': 0.07596940384197835}

=== Starting Run 5/5 ===
[I 2025-10-20 19:40:24,398] A new study created in memory with name: no-name-6db23722-08f0-4853-9c09-827c41a36c0c
[I 2025-10-20 19:40:27,541] Trial 0 finished with value: 0.8724999999999999 and parameters: {'epsilon': 0.9670298390136767, 'learning_rate': 0.5472322491757223}. Best is trial 0 with value: 0.8724999999999999.
[I 2025-10-20 19:40:30,678] Trial 1 finished with value: 0.525 and parameters: {'epsilon': 0.9726843599648843, 'learning_rate': 0.7148159936743647}. Best is trial 0 with value: 0.8724999999999999.
[I 2025-10-20 19:40:33,781] Trial 2 finished with value: 1.5655000000000001 and parameters: {'epsilon': 0.6977288245972708, 'learning_rate': 0.21608949558037638}. Best is trial 2 with value: 1.5655000000000001.
[I 2025-10-20 19:40:36,937] Trial 3 finished with value: 1.703 and parameters: {'epsilon': 0.9762744547762418, 'learning_rate': 0.006230255204589863}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:39,710] Trial 4 finished with value: 0.5325 and parameters: {'epsilon': 0.25298236238344396, 'learning_rate': 0.4347915324044458}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:43,042] Trial 5 finished with value: 1.7025 and parameters: {'epsilon': 0.7793829217937525, 'learning_rate': 0.19768507460025309}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:46,394] Trial 6 finished with value: 0.0 and parameters: {'epsilon': 0.8629932355992223, 'learning_rate': 0.9834006771753128}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:49,121] Trial 7 finished with value: 0.522 and parameters: {'epsilon': 0.16384224140469872, 'learning_rate': 0.5973339439328592}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:51,313] Trial 8 finished with value: 0.0 and parameters: {'epsilon': 0.008986097667554982, 'learning_rate': 0.3865712826436294}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:53,540] Trial 9 finished with value: 0.0 and parameters: {'epsilon': 0.044160057931499574, 'learning_rate': 0.9566529677142359}. Best is trial 3 with value: 1.703.
[I 2025-10-20 19:40:56,811] Trial 10 finished with value: 1.746 and parameters: {'epsilon': 0.8911076130295651, 'learning_rate': 0.1173841955692189}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:41:00,267] Trial 11 finished with value: 0.0 and parameters: {'epsilon': 0.7988111989862254, 'learning_rate': 0.0}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:41:03,904] Trial 12 finished with value: 1.535 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.17147320086034487}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:41:07,600] Trial 13 finished with value: 1.3275000000000001 and parameters: {'epsilon': 0.7787569066370933, 'learning_rate': 0.3179213317579196}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:41:11,141] Trial 14 finished with value: 1.6865 and parameters: {'epsilon': 1.0, 'learning_rate': 0.06198336576877331}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:48:28,690] Trial 15 finished with value: 1.5710000000000002 and parameters: {'epsilon': 0.8830880353763265, 'learning_rate': 0.18459870335728054}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:48:31,135] Trial 16 finished with value: 0.0 and parameters: {'epsilon': 0.0, 'learning_rate': 0.0}. Best is trial 10 with value: 1.746.
[I 2025-10-20 19:48:35,118] Trial 17 finished with value: 1.7215 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.063353915219831}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:26,798] Trial 18 finished with value: 1.7 and parameters: {'epsilon': 0.9411793941709103, 'learning_rate': 0.09845825389620994}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:30,336] Trial 19 finished with value: 0.8710000000000001 and parameters: {'epsilon': 0.5741946430922438, 'learning_rate': 0.5649878389012924}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:34,754] Trial 20 finished with value: 0.6615 and parameters: {'epsilon': 0.45083801172861016, 'learning_rate': 1.0}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:38,382] Trial 21 finished with value: 0.0 and parameters: {'epsilon': 1.0, 'learning_rate': 0.0}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:42,113] Trial 22 finished with value: 1.698 and parameters: {'epsilon': 1.0, 'learning_rate': 0.21618205955331146}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:45,864] Trial 23 finished with value: 1.684 and parameters: {'epsilon': 1.0, 'learning_rate': 0.21037801723325306}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:49,667] Trial 24 finished with value: 1.7114999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.20347430936163938}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:06:53,794] Trial 25 finished with value: 1.6350000000000002 and parameters: {'epsilon': 1.0, 'learning_rate': 0.19658101793434057}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:22:43,115] Trial 26 finished with value: 1.5125 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1900695136818542}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:40:36,604] Trial 27 finished with value: 1.4895 and parameters: {'epsilon': 1.0, 'learning_rate': 0.18875650331405727}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:40:40,085] Trial 28 finished with value: 1.655 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1960477811480977}. Best is trial 10 with value: 1.746.
[I 2025-10-20 20:55:47,499] Trial 29 finished with value: 1.4985 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17055553211680896}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:13:38,637] Trial 30 finished with value: 1.659 and parameters: {'epsilon': 1.0, 'learning_rate': 0.2679429459357729}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:13:42,488] Trial 31 finished with value: 1.5399999999999998 and parameters: {'epsilon': 1.0, 'learning_rate': 0.2566090876026111}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:13:46,243] Trial 32 finished with value: 1.5335 and parameters: {'epsilon': 1.0, 'learning_rate': 0.17872274780997344}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:13:50,163] Trial 33 finished with value: 1.4055 and parameters: {'epsilon': 1.0, 'learning_rate': 0.27955754701601915}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:13:53,564] Trial 34 finished with value: 1.616 and parameters: {'epsilon': 1.0, 'learning_rate': 0.15718335009427056}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:13:57,077] Trial 35 finished with value: 1.5234999999999999 and parameters: {'epsilon': 1.0, 'learning_rate': 0.15603564136551043}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:14:00,534] Trial 36 finished with value: 1.3820000000000001 and parameters: {'epsilon': 1.0, 'learning_rate': 0.16172672426386922}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:14:04,029] Trial 37 finished with value: 0.9894999999999999 and parameters: {'epsilon': 1.0, 'learning_rate': 0.31831829280537116}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:14:07,967] Trial 38 finished with value: 1.6960000000000002 and parameters: {'epsilon': 0.7809417908085313, 'learning_rate': 0.1829553616372224}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:22:00,640] Trial 39 finished with value: 1.7215 and parameters: {'epsilon': 0.7796472982818996, 'learning_rate': 0.17999268181041916}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:22:05,018] Trial 40 finished with value: 1.208 and parameters: {'epsilon': 0.7783348576999982, 'learning_rate': 0.17718491289337954}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:22:08,519] Trial 41 finished with value: 1.7015 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14613481366942513}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:32,563] Trial 42 finished with value: 1.6939999999999997 and parameters: {'epsilon': 1.0, 'learning_rate': 0.14343563004852075}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:36,467] Trial 43 finished with value: 1.3905 and parameters: {'epsilon': 1.0, 'learning_rate': 0.13354918443882555}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:40,448] Trial 44 finished with value: 1.5475 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.15167619707318}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:44,095] Trial 45 finished with value: 1.207 and parameters: {'epsilon': 0.7998317981064228, 'learning_rate': 0.21438986308117808}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:47,963] Trial 46 finished with value: 1.7309999999999999 and parameters: {'epsilon': 1.0, 'learning_rate': 0.1414230042654899}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:51,919] Trial 47 finished with value: 1.575 and parameters: {'epsilon': 1.0, 'learning_rate': 0.13729260182699116}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:55,217] Trial 48 finished with value: 1.6965 and parameters: {'epsilon': 0.9999999999999999, 'learning_rate': 0.13662732238924932}. Best is trial 10 with value: 1.746.
[I 2025-10-20 21:25:58,635] Trial 49 finished with value: 1.4955000000000003 and parameters: {'epsilon': 1.0, 'learning_rate': 0.13224188183992752}. Best is trial 10 with value: 1.746.
Run 5 completed:
  Best value: 1.7460
  Best params: {'epsilon': 0.8911076130295651, 'learning_rate': 0.1173841955692189}"""

pattern = re.compile(r"Trial \d+ finished with value: ([\d.]+)")

extracted_rewards_str = pattern.findall(log_text)

# Convert the captured strings to floats and save them in the initial flat list
trial_rewards_flat = [float(r) for r in extracted_rewards_str]

# --- Restructure the data into 5 subarrays of length 10 ---
trials_per_run = 50
restructured_array = []
num_runs = len(trial_rewards_flat) // trials_per_run # Should be 50 / 10 = 5

for i in range(num_runs):
    # Calculate the start and end index for the current run (sub-array)
    start_index = i * trials_per_run
    end_index = (i + 1) * trials_per_run
    
    # Slice the flat list to get the 10 trials for this run
    run_trials = trial_rewards_flat[start_index:end_index]
    
    # Append the list of 10 trials to the main restructured array
    restructured_array.append(run_trials)

# --- Output and Verification ---

print("--- Extracted and Restructured Trial Rewards ---")
print(f"Total Runs (Sub-arrays): {len(restructured_array)}")
print(f"Length of each Run Array: {len(restructured_array[0]) if restructured_array else 0}")

# Print the restructured array with some formatting for clarity
print("\nRestructured Array (5 Runs x 10 Trials):")
for i, run in enumerate(restructured_array):
    # Format the floats to 4 decimal places for cleaner output
    formatted_run = [f'{r:.4f}' for r in run]
    print(f"Run {i+1}: {formatted_run}")

# The final variable containing the desired structure is 'restructured_array'
print("\nPython Variable 'restructured_array' content (raw format):")
print(restructured_array)
