import pandas as pd
import snappy as sp
import ast

def calc_inv(path, inv_fn, inv_name, verified=False, prog_int=500, save_int=5000):
	# path: a csv file of knots
	# inv_fn: a function that computes a certain knot invariant
	# inv_name: name of the new column in csv file
	# calculate invariants of knots in path using inv_fn,
	# then store the values in a column named inv_name

	df = pd.read_csv(path)
	df[inv_name] = [pd.NA for _ in range(len(df))]
	failed = 0

	for i, code in enumerate(df['snappy_dt_code']):
		knot = sp.Link(code)
		try:
			df.loc[i, inv_name] = inv_fn(knot, verified=verified)
		except:
			try:
				df.loc[i, inv_name] = inv_fn(knot, hp=True, verified=verified)
			except:
				failed += 1
		if i % prog_int == prog_int-1:
			print(f'{i+1} completed\n{failed} failed', flush=True)
		if i % save_int == save_int-1:
			df.to_csv(path, index=False)
			print('results saved', flush=True)

	print(f'{failed} failed in total', flush=True)
	df.to_csv(path, index=False)
	print('done and saved', flush=True)

def calc_inv_hp(path, inv_fn, inv_name):
	# find missing values and try calculation with high precision
	df = pd.read_csv(path)
	for i, inv in enumerate(df[inv_name]):
		if str(inv) == 'nan':
			code = df['snappy_dt_code'][i]
			try:
				df.loc[i, inv_name] = inv_fn(sp.Link(code), hp=True)
			except:
				print(code + ' failed')
	df.to_csv(path, index=False)

def volume(knot, hp=False, verified=False):
	# knot: a single-component snappy.Link
	# return the volume of the knot exterior
	mfld = knot.exterior()
	if hp:
		mfld = mfld.high_precision()
	return float(mfld.volume(verified=verified))

def cusp_area(knot, hp=False, verified=False):
	# knot: a single-component snappy.Link
	# return the area of the unique (maximal) cusp of knot exterior
	mfld = knot.exterior()
	if hp:
		mfld = mfld.high_precision()
	return float(mfld.cusp_areas(verified=verified)[0])

def chern_simons(knot, hp=False):
	# knot: a single-component snappy.Link
	# return the volume of the knot exterior
	mfld = knot.exterior()
	if hp:
		mfld = mfld.high_precision()
	return float(mfld.chern_simons())

def signature(knot):
	return knot.signature()

def calc_knot_inv(path, inv_fn, inv_name, prog_int=500, save_int=5000):
	# calculate algebraic/combinatorial invariants of knots in path using inv_fn,
	# then store the values in a column named inv_name

	df = pd.read_csv(path)
	df[inv_name] = [pd.NA for _ in range(len(df))]
	failed = 0

	for i, code in enumerate(df['snappy_dt_code']):
		knot = sp.Link(code)
		try:
			df.loc[i, inv_name] = inv_fn(knot)
		except:
			failed += 1
		if i % prog_int == prog_int-1:
			print(f'{i+1} completed', flush=True)
			if failed > 0:
				print(f'{failed} failed')
		if i % save_int == save_int-1:
			df.to_csv(path, index=False)
			print('results saved', flush=True)

	print(f'{failed} failed in total', flush=True)
	df.to_csv(path, index=False)
	print('done and saved', flush=True)

def jones_poly(knot):
	# return the Jones polynomial of knot as a dictionary
	return str(knot.jones_polynomial().dict())

def calc_cusp_translations(path, save_int=2000):
	# calculate the longitudinal and (real and imaginary) meridional translations
	# of the cusp, and store in three columns of the csv file (path)

	df = pd.read_csv(path)
	l = len(df)
	if 'real_meridional_translation' not in df.columns:
		df['real_meridional_translation'] = [pd.NA for _ in range(l)]
		df['imaginary_meridional_translation'] = [pd.NA for _ in range(l)]
		df['longitudinal_translation'] = [pd.NA for _ in range(l)]
	failed = 0

	for i, code in enumerate(df['snappy_dt_code']):
		mfld = sp.Link(code).exterior()
		try:
			meri_trans, long_trans = mfld.cusp_translations()[0]
		except:
			try:
				meri_trans, long_trans = mfld.high_precision().cusp_translations()[0]
			except:
				failed += 1
				continue
		df.loc[i,'real_meridional_translation'] = float(meri_trans.real())
		df.loc[i,'imaginary_meridional_translation'] = float(meri_trans.imag())
		df.loc[i,'longitudinal_translation'] = float(long_trans)
		if i % 100 == 99:
			print(f'{i+1} completed\n{failed} failed', flush=True)
		if i % save_int == save_int-1:
			df.to_csv(path, index=False)
			print('results saved')

	print(f'{failed} failed in total', flush=True)
	df.to_csv(path, index=False)
	print('done and saved')

def calc_systole(path, prog_int=100, save_int=500):
	# calculate the complex length of the systole (shortest geodesic) of the knot exterior
	# and store its real and imaginary parts in two columns of the csv file (path)

	df = pd.read_csv(path)
	l = len(df)
	if 'systole_length' not in df.columns:
		df['systole_length'] = [pd.NA for _ in range(l)]
		df['systole_torsion'] = [pd.NA for _ in range(l)]
	failed = 0
	for i, code in enumerate(df['snappy_dt_code']):
		mfld = sp.Link(code).exterior()
		try:
			systole = mfld.length_spectrum_alt(count=1)[0].length
		except:
			try:
				systole = mfld.high_precision().length_spectrum_alt(count=1)[0].length
			except:
				failed += 1
				continue
		df.loc[i,'systole_length'] = float(systole.real())
		df.loc[i,'systole_torsion'] = float(systole.imag())
		if i % prog_int == prog_int-1:
			print(f'{i+1} completed\n{failed} failed', flush=True)
		if i % save_int == save_int-1:
			df.to_csv(path, index=False)
			print('results saved')

	print(f'{failed} failed in total', flush=True)
	df.to_csv(path, index=False)
	print('done and saved')

def calc_torsion_poly(path, prog_int = 100, save_int=5000):
	# calculate the hyperbolic torsion polynomials of the knot exteriors
	# and store the degrees and constant terms in two columns of the csv file (path)

	df = pd.read_csv(path)
	l = len(df)
	if 'torsion_poly_degree' not in df.columns:
		df['torsion_poly_degree'] = [pd.NA for _ in range(l)]
		df['torsion_poly_const_real'] = [pd.NA for _ in range(l)]
		df['torsion_poly_const_imaginary'] = [pd.NA for _ in range(l)]
	failed = 0

	for i, code in enumerate(df['snappy_dt_code']):
		if i < 1264000 or str(df['torsion_poly_degree'][i]) not in {'nan', '<NA>'}:
			continue
		mfld = sp.Link(code).exterior()
		try:
			p = mfld.hyperbolic_torsion()
		except:
			try:
				p = mfld.high_precision().hyperbolic_torsion()
			except:
				failed += 1
				# print(code + ' failed')
				continue
		df.loc[i,'torsion_poly_degree'] = float(p.degree())
		z = p.constant_coefficient()
		df.loc[i,'torsion_poly_const_real'] = float(z.real())
		df.loc[i,'torsion_poly_const_imaginary'] = float(z.imag())
		if i % prog_int == prog_int-1:
			print(f'{i+1} completed\n{failed} failed', flush=True)
		if i % save_int == save_int-1:
			df.astype({'torsion_poly_degree': 'Int64'}).to_csv(path, index=False)
			print('results saved')

	print(f'{failed} failed in total', flush=True)
	df.astype({'torsion_poly_degree': 'Int64'}).to_csv(path, index=False)
	print('done and saved')

def calc_adj_torsion_poly(path, save_int=1000):
	# calculate the hyperbolic torsion polynomials of the knot exteriors
	# and store the degrees and constant terms in two columns of the csv file (path)

	df = pd.read_csv(path)
	l = len(df)
	if 'adj_torsion_poly_degree' not in df.columns:
		df['adj_torsion_poly_degree'] = [pd.NA for _ in range(l)]
		df['adj_torsion_poly_const_real'] = [pd.NA for _ in range(l)]
		df['adj_torsion_poly_const_imaginary'] = [pd.NA for _ in range(l)]
	failed = 0

	for i, code in enumerate(df['snappy_dt_code']):
		mfld = sp.Link(code).exterior()
		try:
			p = mfld.hyperbolic_adjoint_torsion()
		except:
			try:
				p = mfld.high_precision().hyperbolic_adjoint_torsion()
			except:
				failed += 1
				continue
		df.loc[i,'adj_torsion_poly_degree'] = p.degree()
		z = p.constant_coefficient()
		df.loc[i,'adj_torsion_poly_const_real'] = float(z.real())
		df.loc[i,'adj_torsion_poly_const_imaginary'] = float(z.imag())
		if i % 100 == 99:
			print(f'{i+1} completed\n{failed} failed', flush=True)
		if i % save_int == save_int-1:
			df.to_csv(path, index=False)
			print('results saved')

	print(f'{failed} failed in total', flush=True)
	df.to_csv(path, index=False)
	print('done and saved')

def extract_jones_const(from_path, to_path):
	from_df = pd.read_csv(from_path)
	to_df = pd.read_csv(to_path)

	to_df['jones_const'] = [ast.literal_eval(d).get(0,0) for d in from_df['jones_poly']]
	to_df.astype({'jones_const': 'Int64'}).to_csv(to_path, index=False)

