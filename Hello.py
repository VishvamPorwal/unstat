import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import unsatfit as uf
import numpy as np
import copy
from scipy.special import erfc
import plotly.graph_objects as go

LOGGER = get_logger(__name__)

def getoptiontheta(f, bimodal):
    """Get options for theta_s and theta_r"""
    con_q = []
    ini_q = []
    par_theta = []
    if f.cqs == 'max':
        con_q.append([1, max(f.swrc[1])])
    elif f.cqs == 'fix':
        qs = float(f.qsin)
        con_q.append([1, qs])
    else:
        ini_q.append(max(f.swrc[1]))
        par_theta.append('&theta;<sub>s</sub>')
    cqr = f.cqr
    if cqr == 'fix':
        qr = float(f.qrin)
        if qr <= 0 or qr > max(f.swrc[1]):
            qr = 0
        con_q.append([2, qr])
    if cqr == 'fit' or cqr == 'both' and not bimodal:
        ini_q.append(0)
        par_theta.append('&theta;<sub>r</sub>')
    if cqr == 'both' and bimodal:
        con_q.append([2, 0])
    return con_q, ini_q, par_theta

def model(ID):
    """Define model"""
    q = 'Q(x) &=& \mathrm{erfc}(x/\sqrt{2})/2'
    # q = 'Q(x) &=& \mathrm{erfc}(x/\sqrt{2})/2 = \int_{x}^{\infty}\\frac{\exp(-x^2/2)}{\sqrt{2\pi}}dx'
    if ID == 'all':
        return model('unimodal') + model('bimodal')
    if ID == 'unimodal':
        return ('BC', 'VG', 'KO', 'FX')
    if ID == 'bimodal':
        return ('DBCH', 'VGBCCH', 'DVCH', 'KOBCCH', 'DB', 'DV', 'DK')
    if ID == 'limit':
        return ('max_qs', 'max_lambda_i', 'max_n_i', 'min_sigma_i')
    if ID == 'savekeys':
        return model('all') + model('limit') + ('onemodel', 'cqs', 'cqr', 'qsin', 'qrin', 'input')
    if ID == 'BC':
        return {
            'html': 'Brooks and Corey',
            'label': 'BC',
            'equation': 'S_e = \\begin{cases}\left(h / h_b\\right)^{-\lambda} & (h>h_b) \\\\ 1 & (h \le h_b)\end{cases}',
            'parameter': ('h<sub>b</sub>', '&lambda;'),
            'note': '',
            'selected': True
        }
    if ID == 'VG':
        return {
            'html': 'van Genuchten',
            'label': 'VG',
            'equation': 'S_e = \\biggl[\dfrac{1}{1+(\\alpha h)^n}\\biggr]^m ~~ (m=1-1/n)',
            'parameter': ('&alpha;', 'n'),
            'note': '',
            'selected': True
        }
    if ID == 'KO':
        return {
            'html': 'Kosugi',
            'label': 'KO',
            'equation': '\\begin{eqnarray}S_e &=& Q \\biggl[\dfrac{\ln(h/h_m)}{\sigma}\\biggr]\\\\' + q + '\end{eqnarray}',
            'parameter': ('h<sub>m</sub>', '&sigma;'),
            'note': '',
            'selected': False
        }
    if ID == 'FX':
        return {
            'html': 'Fredlund and Xing',
            'label': 'FX',
            'equation': 'S_e = \\biggl[ \dfrac{1}{\ln \left[e+(h / a)^n \\right]} \\biggr]^m',
            'parameter': ('a', 'm', 'n'),
            'note': 'For FX model, e is Napier\'s constant.',
            'selected': False
        }
    if ID == 'DBCH':
        return {
            'html': 'dual-BC-CH',
            'label': 'dual-BC-CH',
            'equation': 'S_e = \\begin{cases}w_1 \left(h / h_b\\right)^{-\lambda_1} + (1-w_1)\left(h / h_b\\right)^{-\lambda_2}  & (h>h_b)\\\\ 1 & (h \le h_b)\end{cases}',
            'parameter': ('w<sub>1</sub>', 'h<sub>b</sub>', '&lambda;<sub>1</sub>', '&lambda;<sub>2</sub>'),
            'note': '',
            'selected': False
        }
    if ID == 'VGBCCH':
        return {
            'html': 'VG<sub>1</sub>BC<sub>2</sub>-CH',
            'label': '$\mathrm{VG}_1\mathrm{BC}_2$-CH',
            'equation': '\\begin{eqnarray}S_e &=& \\begin{cases}w_1 S_1 + (1-w_1)\left(h/H\\right)^{-\lambda_2}  & (h>H)\\\\ w_1 S_1 + 1-w_1 & (h \le H)\end{cases}\\\\S_1 &=& \\bigl[1+(h/H)^{n_1}\\bigr]^{-{m_1}} ~~ (m_1=1-1/{n_1})\end{eqnarray}',
            'parameter': ('w<sub>1</sub>', 'H', 'n<sub>1</sub>', '&lambda;<sub>2</sub>'),
            'note': '',
            'selected': False
        }
    if ID == 'DVCH':
        return {
            'html': 'dual-VG-CH',
            'label': 'dual-VG-CH',
            'equation': '\\begin{eqnarray}S_e &=& w_1\\bigl[1+(\\alpha h)^{n_1}\\bigr]^{-m_1} + (1-w_1)\\bigl[1+(\\alpha h)^{n_2}\\bigr]^{-m_2}\\\\m_i&=&1-1/{n_i}\end{ }',
            'parameter': ('w<sub>1</sub>', '&alpha;', 'n<sub>1</sub>', 'n<sub>2</sub>'),
            'note': '',
            'selected': True
        }
    if ID == 'KOBCCH':
        return {
            'html': 'KO<sub>1</sub>BC<sub>2</sub>-CH',
            'label': '$\mathrm{KO}_1\mathrm{BC}_2$-CH',
            'equation': '\\begin{eqnarray}S_e &=& \\begin{cases}w_1 S_1 + (1-w_1)\left(h/H\\right)^{-\lambda_2}  & (h>H)\\\\ w_1 S_1 + 1-w_1 & (h \le H)\end{cases}\\\\S_1 &=& Q \\biggl[\dfrac{\ln(h/h_m)}{\sigma_1}\\biggr], Q(x) = \mathrm{erfc}(x/\sqrt{2})/2\end{eqnarray}',
            'parameter': ('w<sub>1</sub>', 'H', '&sigma;<sub>1</sub>', '&lambda;<sub>2</sub>'),
            'note': '',
            'selected': True
        }
    if ID == 'DB':
        return {
            'html': 'dual-BC',
            'label': 'dual-BC',
            'equation': 'S_e = \\begin{cases}w_1 \left(h / h_{b_1}\\right)^{-\lambda_1} + (1-w_1)\left(h / h_{b_2}\\right)^{-\lambda_2}  & (h>h_{b_2}) \\\\ ' +
            'w_1 \left(h / h_{b_1}\\right)^{-\lambda_1} + 1-w_1  & (h_{b_1} < h \le h_{b_2}) \\\\1 & (h \le h_{b_1})\end{cases}',
            'parameter': ('w<sub>1</sub>', 'hb<sub>1</sub>', '&lambda;<sub>1</sub>', 'hb<sub>2</sub>', '&lambda;<sub>2</sub>'),
            'note': '',
            'selected': False
        }
    if ID == 'DV':
        return {
            'html': 'dual-VG',
            'label': 'dual-VG',
            'equation': '\\begin{eqnarray}S_e &=& w_1\\bigl[1+(\\alpha_1 h)^{n_1}\\bigr]^{-m_1} + (1-w_1)\\bigl[1+(\\alpha_2 h)^{n_2}\\bigr]^{-m_2}\\\\m_i&=&1-1/{n_i}\end{eqnarray}',
            'parameter': ('w<sub>1</sub>', '&alpha;<sub>1</sub>', 'n<sub>1</sub>', '&alpha;<sub>2</sub>', 'n<sub>2</sub>'),
            'note': '',
            'selected': True
        }
    if ID == 'DK':
        return {
            'html': 'dual-KO',
            'label': 'dual-KO',
            'equation': '\\begin{eqnarray}S_e &=& w_1 Q \\biggl[\dfrac{\ln(h/h_{m_1})}{\sigma_1}\\biggr] + (1-w_1) Q \\biggl[\dfrac{\ln(h/h_{m_2})}{\sigma_2}\\biggr]\\\\' + q + '\end{eqnarray}',
            'parameter': ('w<sub>1</sub>', 'hm<sub>1</sub>', '&sigma;<sub>1</sub>', 'hm<sub>2</sub>', '&sigma;<sub>2</sub>'),
            'note': '',
            'selected': False
        }

def bc_pred(h, hb, lamda):
    if h > hb:
        Se = (h / hb) ** -lamda
    else:
        Se = 1
    return Se

def vg_pred(h, alpha, n):
    m = 1 - 1 / n
    Se = (1 + (alpha * h) ** n) ** -m
    return Se

def ko_pred(h, hm, sigma):
    def Q(x):
        return erfc(x / np.sqrt(2)) / 2

    Se = Q(np.log(h / hm) / sigma)
    return Se

def fx_pred(h, a, m, n):
    Se = 1 / np.log(np.e + (h / a) ** n) ** m
    return Se

def dbch_pred(h, w1, hb, lamda1, lamda2):
    if h > hb:
        Se = w1 * (h / hb) ** -lamda1 + (1 - w1) * (h / hb) ** -lamda2
    else:
        Se = 1
    return Se

def vgbcch_pred(h, w1, H, n1, lamda2):
    m1 = 1 - 1 / n1
    S1 = (1 + (h / H) ** n1) ** -m1
    if h > H:
        Se = w1 * S1 + (1 - w1) * (h / H) ** -lamda2
    else:
        Se = w1 * S1 + 1 - w1
    return Se

def dvch_pred(h, w1, alpha, n1, n2):
    m1 = 1 - 1 / n1
    m2 = 1 - 1 / n2
    Se = w1 * (1 + (alpha * h) ** n1) ** -m1 + (1 - w1) * (1 + (alpha * h) ** n2) ** -m2
    return Se

def kobcch_pred(h, w1, H, hm, sigma1, lamda2):
    def Q(x):
        return erfc(x / np.sqrt(2)) / 2

    S1 = Q(np.log(h / hm) / sigma1)
    if h > H:
        Se = w1 * S1 + (1 - w1) * (h / H) ** -lamda2
    else:
        Se = w1 * S1 + 1 - w1
    return Se

def db_pred(h, w1, hb1, hb2, lamda1, lamda2):
    if h > hb2:
        Se = w1 * (h / hb1) ** -lamda1 + (1 - w1) * (h / hb2) ** -lamda2
    elif hb1 < h <= hb2:
        Se = w1 * (h / hb1) ** -lamda1 + 1 - w1
    else:
        Se = 1
    return Se

def dv_pred(h, w1, alpha1, alpha2, n1, n2):
    m1 = 1 - 1 / n1
    m2 = 1 - 1 / n2
    Se = w1 * (1 + (alpha1 * h) ** n1) ** -m1 + (1 - w1) * (1 + (alpha2 * h) ** n2) ** -m2
    return Se

def dk_pred(h, w1, hm1, sigma1, hm2, sigma2):
    def Q(x):
        return erfc(x / np.sqrt(2)) / 2

    S1 = Q(np.log(h / hm1) / sigma1)
    S2 = Q(np.log(h / hm2) / sigma2)
    Se = w1 * S1 + (1 - w1) * S2
    return Se


def run():
    st.set_page_config(
        page_title="Unsatfit",
    )
    file_option = st.radio(
        "Choose input method",
        ("Upload file", "Use demo file")
    )

    flag = False
    flag_rename = False

    if file_option == "Upload file":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is None:
            st.write("Please upload a CSV or Excel file.")
        else:    
            if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                # Read Excel file
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                flag = True
            else:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                flag = True
            st.write(df)
    else:
        # Load demo file from the directory
        demo_file_path = 'swc_dat_in.csv' 
        df = pd.read_csv(demo_file_path)
        st.write("Demo Data")
        st.write(df)
        flag = True
        flag_rename = True

    if file_option == "Upload file" and flag:

        columns = df.columns
        # add placeholder to columns
        columns = ['choose column'] + columns.tolist()
        # select which columns are id, mbar, swc
        id_col = st.selectbox('Select id column', columns)
        mbar_col = st.selectbox('Select mbar column', columns)
        swc_col = st.selectbox('Select swc column', columns)

        if id_col != 'id' and id_col != 'choose column':
            df.rename(columns={id_col: 'id'}, inplace=True)
        if mbar_col != 'mbar' and mbar_col != 'choose column':
            df.rename(columns={mbar_col: 'mbar'}, inplace=True)
        if swc_col != 'swc' and swc_col != 'choose column':
            df.rename(columns={swc_col: 'swc'}, inplace=True)

        if id_col != 'choose column' and mbar_col != 'choose column' and swc_col != 'choose column':
            flag_rename = True

    if not flag_rename:
        st.write('Please rename the columns')
        return
    
    # warn and drop NaN rows
    if df.isnull().values.any():
        st.warning('There are NaN values in the data. These rows will be dropped.')
        df = df.dropna()

    
    if flag and flag_rename:
        models = ['BC', 'KO', 'DBCH', 'VG', 'VGBCCH', 'FX', 'DVCH', 'KOBCCH', 'DB', 'DV', 'DK']
            # select models multiple
        selected_models = st.multiselect('Select models', models)
            

        if len(selected_models) == 0:
            st.write('Please select at least one model')
            return
        result = []
        # group by "id" and show
        df_grouped = df.groupby('id')
        for name, group in df_grouped:
            try: 
                f = uf.Fit()
                h_t = group['mbar'].values
                theta = group['swc'].values
                f = uf.Fit()
                f.cqr = 'both'
                f.cqs = 'fit'
                f.max_qs = 1.5
                f.max_lambda_i = 10
                f.max_n_i = 8
                f.swrc = (h_t, theta)
                f.min_sigma_i = 0.2
                f.selectedmodel = selected_models

                f.show_fig = False
                f.save_fig = True
                f.filename = 'img/swrc.png'
                f.fig_width = 5.5  # inch
                f.fig_height = 4.5
                f.top_margin = 0.05
                f.bottom_margin = 0.12
                f.left_margin = 0.15  # Space for label is needed
                f.right_margin = 0.05
                f.legend_loc = 'upper right'
                f.color_marker = 'blue'

                con_q, ini_q, par_theta = getoptiontheta(f, False)


                # BC (Brooks and Corey) model
                f.b_qs = b_qs = (0, max(f.swrc[1]) * f.max_qs)
                f.b_lambda1 = f.b_lambda2 = b_lambda_i = (0, f.max_lambda_i)
                f.b_m = f.b_sigma = (0, np.inf)
                max_m_i = 1 - 1/f.max_n_i



                if 'BC' in f.selectedmodel:
                    f.set_model('BC', const=[*con_q])
                    hb, l = f.get_init()  # Get initial parameter
                    f.ini = (*ini_q, hb, l)
                    f.optimize()
                    if not f.success:
                        f.b_qs = (0, max(f.swrc[1])*min(1.05, f.max_qs))
                        f.optimize()
                        f.b_qs = b_qs
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        # f.optimize
                        if not f.success:
                            f = copy.deepcopy(f2)
                    f.fitted_show = f.fitted
                    f.setting = model('BC')
                    f.par = (*par_theta, *f.setting['parameter'])
                    f2 = copy.deepcopy(f)

                    result.append({"id": name, 'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht, "model": "BC", 'parameter_names': f2.par, 'parameter_values': f2.fitted})

                # VG (van Genuchten) model
                f.set_model('VG', const=[*con_q, 'q=1'])
                a, m = f.get_init()  # Get initial parameter
                f.ini = (*ini_q, a, m)
                f.optimize()
                q = f.fitted[:-2]
                a, m = f.fitted[-2:]
                n = 1/(1-m)
                vg_r2 = f.r2_ht
                f.fitted_show = [*f.fitted[:-1], n]  # Convert from m to n
                f.setting = model('VG')
                f.par = (*par_theta, *f.setting['parameter'])
                if 'VG' in f.selectedmodel:
                    f2 = copy.deepcopy(f)
                    result.append({"id": name,'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht,  "model": "VG", 'parameter_names': f2.par, 'parameter_values': f2.fitted})

                # KO (Kosugi) model
                if 'KO' in f.selectedmodel or 'FX' in f.selectedmodel:
                    f.set_model('KO', const=[*con_q])
                    sigma = 1.2*(n-1)**(-0.8)
                    f.ini = (*q, 1/a, sigma)
                    f.optimize()
                    if not f.success or f.r2_ht < vg_r2 - 0.1:
                        hb, l = f.get_init_bc()
                        sigma = 1.2*l**(-0.8)
                        if sigma > 2.5:
                            sigma = 2.5
                        f.ini = (*ini_q, hb, sigma)
                        f.b_qs = (0, max(f.swrc[1])*min(1.05, f.max_qs))
                        f.optimize()
                        f.b_qs = b_qs
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        # f.optimize
                        if not f.success:
                            f = copy.deepcopy(f2)
                    q_ko = f.fitted[:-2]
                    if f.success:
                        hm, sigma = f.fitted[-2:]
                        ko_r2 = f.r2_ht
                    else:
                        ko_r2 = 0
                    if 'KO' in f.selectedmodel:
                        f.setting = model('KO')
                        f.fitted_show = f.fitted
                        f.par = (*par_theta, *f.setting['parameter'])
                        f2 = copy.deepcopy(f)
                        result.append({"id": name, 'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht, "model": "KO", 'parameter_names': f2.par, 'parameter_values': f2.fitted})

                # FX (Fredlund and Xing) model
                if 'FX' in f.selectedmodel:
                    f.set_model('FX', const=[*con_q])
                    if vg_r2 > ko_r2:
                        f.ini = (*q, 1/a, 2.54 * (1-1/n), 0.95 * n)
                    else:
                        f.ini = (*q_ko, hm, 2.54, 1.52 / sigma)
                    f.optimize()
                    if not f.success:
                        hb, l = f.get_init_bc()
                        n = l + 1
                        a, m, n = hb, 2.54 * (1-1/n), 0.95 * n
                        f.b_qs = (0, max(f.swrc[1]))
                        f.b_qr = (0, min(f.swrc[1])/2)
                        f.ini = (*ini_q, a, m, n)
                        f.optimize()
                        f.b_qs = b_qs
                        f.b_qr = (0, np.inf)
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        # f.optimize
                        if not f.success:
                            f = copy.deepcopy(f2)
                    f.setting = model('FX')
                    f.fitted_show = f.fitted
                    f.par = (*par_theta, *f.setting['parameter'])
                    f2 = copy.deepcopy(f)
                    result.append({"id": name, 'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht,  "model": "FX", 'parameter_names': f2.par, 'parameter_values': f2.fitted})

                
                # Bimodal model
                if any(name in f.selectedmodel for name in model('bimodal')):
                    con_q, ini_q, par_theta = getoptiontheta(f, True)
                f.b_m = (0, max_m_i)

                # dual-BC-CH model
                if 'DBCH' in f.selectedmodel or 'VGBCCH' in f.selectedmodel or 'DB' in f.selectedmodel or 'KOBCCH' in f.selectedmodel:
                    f.set_model('dual-BC-CH', const=[*con_q])
                    hb, hc, l1, l2 = f.get_init()
                    if l1 > f.max_lambda_i:
                        l1 = f.max_lambda_i - 0.00001
                    if l2 > f.max_lambda_i:
                        l2 = f.max_lambda_i - 0.00001
                    f.ini = (*ini_q, hb, hc, l1, l2)  # Get initial parameter
                    f.optimize()
                    if not f.success or f.r2_ht < vg_r2 - 0.05:
                        hb2, l1 = f.get_init_bc()
                        l2 = l/5
                        if l1 > f.max_lambda_i:
                            l1 = f.max_lambda_i - 0.00001
                        if l2 > f.max_lambda_i:
                            l2 = f.max_lambda_i - 0.00001
                        f.ini = (*ini_q, hb, hb*2, l1, l2)
                        f.optimize()
                        if not f.success:
                            f.b_qs = (max(f.swrc[1]) * 0.9, max(f.swrc[1]))
                            f.b_qr = (0, min(f.swrc[1])/100)
                            f.b_lambda1 = f.b_lambda2 = (0, l*1.5)
                            f.ini = (*ini_q, hb, hb, l, l)
                            f.optimize()
                            f.b_qs = b_qs
                            f.b_qr = (0, np.inf)
                            f.b_lambda1 = f.b_lambda2 = b_lambda_i
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        # f.optimize
                        if not f.success:
                            f = copy.deepcopy(f2)
                    if f.success:
                        hb, hc, l1, l2 = f.fitted[-4:]
                        w1 = 1/(1+(hc/hb)**(l2-l1))
                        q = f.fitted[:-4]
                        f.fitted_show = (*q, w1, hb, l1, l2)
                    f.setting = model('DBCH')
                    f.par = (*par_theta, *f.setting['parameter'])
                    dbch = copy.deepcopy(f)
                    if 'DBCH' in f.selectedmodel:
                        result.append({"id": name, 'r2_ht': dbch.r2_ht, 'aic_ht': dbch.aic_ht,  "model": "DBCH", 'parameter_names': dbch.par, 'parameter_values': dbch.fitted})


                    # VG1BC2-CH model
            
                if 'VGBCCH' in f.selectedmodel:
                    f.set_model('VG1BC2-CH', const=[*con_q, 'q=1'])
                    if dbch.success:
                        n1 = l1 + 1
                        m1 = 1-1/n1
                        if m1 < 0.1:
                            m1 = 0.1
                        if m1 > 0.8:
                            m1 = 0.8
                        f.ini = (*q, w1, 1/hb, m1, l2)
                        f.optimize()
                        if not f.success:
                            f.b_qs = (max(f.swrc[1]) * 0.95,
                                    max(f.swrc[1]) * min(1.05, f.max_qs))
                            f.b_qr = (0, min(f.swrc[1]) / 10)
                            f.ini = (*ini_q, w1, 1/hb, m, l2)
                            f.optimize()
                            if not f.success:
                                f.ini = (*ini_q, 0.9, 1/hb, m, l2)
                                f.b_lambda2 = (l2 * 0.8, l2 * 1.2)
                                f.optimize()
                            f.b_qs = b_qs
                            f.b_qr = (0, np.inf)
                            f.b_lambda2 = b_lambda_i
                            f2 = copy.deepcopy(f)
                            f.ini = f.fitted
                            # f.optimize
                            if not f.success:
                                f = copy.deepcopy(f2)
                    else:
                        f.ini = (*ini_q, 0.9, a, m, m/2)
                        f.optimize()
                    if f.success:
                        w1, a1, m1, l2 = f.fitted[-4:]
                        n1 = 1/(1-m1)
                        q = f.fitted[:-4]
                        f.fitted_show = (*q, w1, 1/a1, n1, l2)
                    f.setting = model('VGBCCH')
                    f.par = (*par_theta, *f.setting['parameter'])
                    vgbcch = copy.deepcopy(f)
                    result.append({"id": name,'r2_ht': vgbcch.r2_ht, 'aic_ht': vgbcch.aic_ht,  "model": "VGBCCH", 'parameter_names': vgbcch.par, 'parameter_values': vgbcch.fitted})
                # dual-VG-CH model
                if 'DVCH' in f.selectedmodel:
                    f.set_model('dual-VG-CH', const=[*con_q, 'q=1'])
                    w1, a, m1, m2 = f.get_init()  # Get initial parameter
                    if m1 > max_m_i:
                        m1 = max_m_i - 0.0001
                    if m2 > max_m_i:
                        m2 = max_m_i - 0.0001
                    f.ini = (*ini_q, w1, a, m1, m2)
                    f.optimize()
                    if not f.success:
                        f.b_qs = (max(f.swrc[1]) * 0.95,
                                max(f.swrc[1]) * min(1.05, f.max_qs))
                        f.b_m = (0, 0.9)
                        a, m = f.get_init_vg()
                        if m > 0.9:
                            m = 0.9
                        f.ini = (*ini_q, 0.5, a, m, m)
                        f.optimize()
                        f.b_qs = b_qs
                        f.b_m = (0, max_m_i)
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        # f.optimize
                        if not f.success:
                            f = copy.deepcopy(f2)
                    if f.success:
                        w1, a1, m1, m2 = f.fitted[-4:]
                        q = f.fitted[:-4]
                        n1 = 1/(1-m1)
                        n2 = 1/(1-m2)
                        f.fitted_show = (*q, w1, a1, n1, n2)
                    f.setting = model('DVCH')
                    f.par = (*par_theta, *f.setting['parameter'])
                    dvch = copy.deepcopy(f)
                    if 'DVCH' in f.selectedmodel:
                        result.append({"id": name, 'r2_ht': dvch.r2_ht, 'aic_ht': dvch.aic_ht,  "model": "DVCH", 'parameter_names': dvch.par, 'parameter_values': dvch.fitted})

                # KO1BC2-CH model
                if 'KOBCCH' in f.selectedmodel:
                    f.set_model('KO1BC2-CH', const=[*con_q])
                    f.b_sigma = (f.min_sigma_i, np.inf)
                    if dbch.success:
                        s1 = 1.2*l1**(-0.8)
                        if s1 > 2:
                            s1 = 2
                        if s1 < f.min_sigma_i:
                            s1 = f.min_sigma_i + 0.00001
                        if l2 > f.max_lambda_i:
                            l2 = f.max_lambda_i - 0.00001
                        f.ini = (*q, w1, hb, s1, l2)
                        f.optimize()
                        if not f.success:
                            f.b_qs = (max(f.swrc[1]) * 0.95,
                                    max(f.swrc[1]) * min(1.05, f.max_qs))
                            f.b_qr = (0, min(f.swrc[1]) / 10)
                            f.ini = (*ini_q, w1, hb, s1, l2)
                            if s1 < f.min_sigma_i:
                                s1 = f.min_sigma_i + 0.00001
                            if l2 > f.max_lambda_i:
                                l2 = f.max_lambda_i - 0.00001
                            f.optimize()
                            if not f.success:
                                f.ini = (*ini_q, 0.9, hb, s1, l2)
                                f.b_lambda2 = (l2 * 0.8, min(l2 * 1.2, f.max_lambda_i))
                                f.optimize()
                            f.b_qs = b_qs
                            f.b_qr = (0, np.inf)
                            f.b_lambda2 = b_lambda_i
                            f2 = copy.deepcopy(f)
                            f.ini = f.fitted
                            # f.optimize
                            if not f.success:
                                f = copy.deepcopy(f2)
                    else:
                        w1, hm, sigma1, l2 = f.get_init_kobcch()
                        if l2 > f.max_lambda_i:
                            l2 = f.max_lambda_i - 0.00001
                        f.ini = (*ini_q, w1, hm, sigma1, l2)
                        f.optimize()
                    if f.success:
                        w1, hm, s1, l2 = f.fitted[-4:]
                        q = f.fitted[:-4]
                        f.fitted_show = (*q, w1, hm, s1, l2)
                    f.setting = model('KOBCCH')
                    f.par = (*par_theta, *f.setting['parameter'])
                    vgbcch = copy.deepcopy(f)
                    result.append({"id": name, 'r2_ht': vgbcch.r2_ht, 'aic_ht': vgbcch.aic_ht,  "model": "KOBCCH", 'parameter_names': vgbcch.par, 'parameter_values': vgbcch.fitted})

                # dual-BC model
                if 'DB' in f.selectedmodel:
                    f.set_model('dual-BC', const=[*con_q])
                    if dbch.success:
                        hb, hc, l1, l2 = dbch.fitted[-4:]
                        w1 = 1/(1+(hc/hb)**(l2-l1))
                        q = dbch.fitted[:-4]
                        f.ini = (*ini_q, w1, hb*0.9, l1, (hb*max(f.swrc[0]))**0.5, l2)
                        f.b_qs = (max(f.swrc[1]) * 0.95,
                                max(f.swrc[1]) * min(1.05, f.max_qs))
                        f.optimize()
                        if not f.success:
                            f.b_qr = (0, min(f.swrc[1]) / 10)
                            f.ini = (*ini_q, w1, hb*0.9, l1, hb*1.1, l2)
                            f.optimize()
                            if not f.success:
                                hb, l = f.get_init_bc()
                                if l > f.max_lambda_i:
                                    l = f.max_lambda_i - 0.00001
                                f.b_lambda1 = f.b_lambda2 = (
                                    0, min(l * 1.1, f.max_lambda_i))
                                f.ini = (*ini_q, 0.7, hb, l, hb, l)
                                f.optimize()
                            f.b_qr = (0, np.inf)
                            f.b_lambda1 = f.b_lambda2 = b_lambda_i
                            f2 = copy.deepcopy(f)
                            f.ini = f.fitted
                            # f.optimize
                            if not f.success or f.r2_ht < f2.r2_ht:
                                f = copy.deepcopy(f2)
                        f.b_qs = b_qs
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        # f.optimize
                        if not f.success or f.r2_ht < f2.r2_ht:
                            f = copy.deepcopy(f2)
                    else:
                        hb, l = f.get_init_bc()
                        if l > f.max_lambda_i:
                            l = f.max_lambda_i - 0.00001
                        f.ini = (*ini_q, 0.7, hb, l, hb, l)
                        f.optimize()
                    if f.success:
                        w1, hb1, l1, hb2, l2 = f.fitted[-5:]
                        q = f.fitted[:-5]
                        if hb1 > hb2:
                            hb1, hb2 = hb2, hb1
                            l1, l2 = l2, l1
                            w1 = 1-w1
                            f.fitted = (*q, w1, hb1, l1, hb2, l2)
                        f.fitted_show = f.fitted
                    f.setting = model('DB')
                    f.par = (*par_theta, *f.setting['parameter'])
                    f2 = copy.deepcopy(f)
                    result.append({"id": name, 'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht, "model": "DB", 'parameter_names': f2.par, 'parameter_values': f2.fitted})

                # dual-VG model
                if 'DV' in f.selectedmodel or 'DK' in f.selectedmodel:
                    f.set_model('dual-VG', const=[*con_q, 'q=1'])
                    w1, a1, m1, a2, m2 = init_vg2 = f.get_init_vg2()
                    if m1 > max_m_i:
                        m1 = max_m_i - 0.0001
                    if m2 > max_m_i:
                        m2 = max_m_i - 0.0001
                    f.ini = (*ini_q, w1, a1, m1, a2, m2)
                    f.optimize()
                    vg2_r2 = 0
                    if f.success:
                        vg2_r2 = f.r2_ht
                        w1, a1, m1, a2, m2 = f.fitted[-5:]
                        q = f.fitted[:-5]
                        if a1 < a2:
                            a1, a2 = a2, a1
                            m1, m2 = m2, m1
                            w1 = 1-w1
                            f.fitted = (*q, w1, a1, m1, a2, m2)
                        n1 = 1/(1-m1)
                        n2 = 1/(1-m2)
                        f.fitted_show = (*q, w1, a1, n1, a2, n2)
                    f.setting = model('DV')
                    f.par = (*par_theta, *f.setting['parameter'])
                    f2 = copy.deepcopy(f)
                    if 'DV' in f.selectedmodel:
                        result.append({"id": name, 'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht,  "model": "DV", 'parameter_names': f2.par, 'parameter_values': f2.fitted})

                # dual-KO model
                if 'DK' in f.selectedmodel:
                    if f.success:
                        if n1 < 1.4:
                            n1 = 1.4
                        s1 = 1.2*(n1-1)**(-0.8)
                        if s1 < f.min_sigma_i:
                            s1 = f.min_sigma_i + 0.00001
                        if n2 < 1.4:
                            n2 = 1.4
                        s2 = 1.2*(n2-1)**(-0.8)
                        if s2 < f.min_sigma_i:
                            s2 = f.min_sigma_i + 0.00001
                        if s2 > 2:
                            s2 = 2
                        hm1 = 1/a1
                        hm2 = 1/a2
                        f.set_model('dual-KO', const=[*con_q])
                        f.b_w1 = (max(w1 * 0.5, w1-0.15), min(w1+0.15, 1-(1-w1)*0.5))
                        f.b_hm1 = (hm1 / 8, hm1 * 8)
                        f.b_hm2 = (min(max(f.swrc[0])*0.5, hm2 / 8), hm2 * 8)
                        f.b_sigma = (f.min_sigma_i, 2.5)
                        f.ini = (*q, w1, hm1, s1, hm2, s2)
                        f.optimize()
                        f.b_w1 = (0, 1)
                        f.b_hm1 = f.b_hm2 = (0, np.inf)
                        if not f.success or f.r2_ht < vg2_r2 - 0.05:
                            f.b_qs = (max(f.swrc[1]) * 0.95,
                                    max(f.swrc[1]) * min(1.05, f.max_qs))
                            f.b_qr = (0, min(f.swrc[1]) / 10)
                            f.ini = (*ini_q, w1, 1/a1, s1, 1/a2, s2)
                            f.optimize()
                            if not f.success:
                                a, m = f.get_init_vg()
                                n = 1/(1-m)
                                if n < 1.4:
                                    n = 1.4
                                s = 1.2*(n-1)**(-0.8)
                                f.ini = (*ini_q, 0.5, 1/a, s, 1/a, s)
                                f.optimize()
                            f.b_qs = b_qs
                            f.b_qr = (0, np.inf)
                            f2 = copy.deepcopy(f)
                            f.ini = f.fitted
                            f.optimize()
                            if not f.success:
                                f = copy.deepcopy(f2)
                        f.b_sigma = (f.min_sigma_i, np.inf)
                        f2 = copy.deepcopy(f)
                        f.ini = f.fitted
                        f.optimize()
                        if not f.success or f.r2_ht < f2.r2_ht:
                            f = copy.deepcopy(f2)
                        if f.success:
                            w1, hm1, s1, hm2, s2 = f.fitted[-5:]
                            q = f.fitted[:-5]
                            if hm1 > hm2:
                                hm1, hm2 = hm2, hm1
                                s1, s2 = s2, s1
                                w1 = 1-w1
                                f.fitted = (*q, w1, hm1, s1, hm2, s2)
                            f.fitted_show = f.fitted
                    f.setting = model('DK')
                    f.par = (*par_theta, *f.setting['parameter'])
                    f2 = copy.deepcopy(f)
                    result.append({"id": name,'r2_ht': f2.r2_ht, 'aic_ht': f2.aic_ht,  "model": "DK", 'parameter_names': f2.par, 'parameter_values': f2.fitted})
            except Exception as e:
                st.write(f"Error in fitting for id: {name}")
                continue

        # show results
        result_df = pd.DataFrame(result)
        # group by "id" and "model"
        df_grouped = result_df.groupby(['id'])

        aics = {}
        model_results = {}
        # download by models
        for mod in selected_models:
            result_df_mod = result_df[result_df['model'] == mod]
            result_df_mod = result_df_mod.drop(columns=['model'])
            pns = []
            for pn in result_df_mod['parameter_names'].values[0]:
                pns.append(pn)
            # columns for each parameter
            result_df_mod = result_df_mod.drop(columns=['parameter_names'])
            for i, pn in enumerate(pns):
                result_df_mod[pn] = result_df_mod['parameter_values'].apply(lambda x: x[i])
            result_df_mod = result_df_mod.drop(columns=['parameter_values'])
            # reset index
            result_df_mod = result_df_mod.reset_index(drop=True)
            # st.markdown(f"## {mod}")
            # st.write(result_df_mod)
            model_results[mod] = result_df_mod
            # Calculate mean of aic_ht
            aic_mean = result_df_mod['aic_ht'].mean()
            aics[mod] = aic_mean
        # TODO: one more loop, first print best model and then give different model results
        # best model with least mean AIC
        best_model = min(aics, key=aics.get)
        st.markdown(f"# Best model: {best_model}, Mean AIC: {aics[best_model]}")

        # show best model results
        st.markdown(f"## {best_model}")
        st.write(model_results[best_model])

        # show other models results on demand
        if len(selected_models) > 1:
            demand = st.checkbox("Show other models results")
            if demand:
                for mod in selected_models:
                    if mod != best_model:
                        st.markdown(f"## {mod}")
                        st.write(model_results[mod])

        # predict Se for the best model, h = 0.00001 to 15000 in 100 intervals
        h = np.linspace(100, 15000, 100)
        # add 330 in h
        h = np.append(h, 330)
        h = np.append(h, 0.00001)
        h = np.sort(h)

        results_best_model = {}
        for index, row in model_results[best_model].iterrows():
            for i in range(len(h)):
                if best_model == 'BC':
                    se = bc_pred(h[i], *row[-2:])
                elif best_model == 'KO':
                    se = ko_pred(h[i], *row[-2:])
                elif best_model == 'DBCH':
                    se = dbch_pred(h[i], *row[-4:])
                elif best_model == 'VG':
                    se = vg_pred(h[i], *row[-2:])
                elif best_model == 'VGBCCH':
                    se = vgbcch_pred(h[i], *row[-4:])
                elif best_model == 'FX':
                    se = fx_pred(h[i], *row[-3:])
                elif best_model == 'DVCH':
                    se = dvch_pred(h[i], *row[-4:])
                elif best_model == 'KOBCCH':
                    se = kobcch_pred(h[i], *row[-5:])
                elif best_model == 'DB':
                    se = db_pred(h[i], *row[-5:])
                elif best_model == 'DV':
                    se = dv_pred(h[i], *row[-5:])
                elif best_model == 'DK':
                    se = dk_pred(h[i], *row[-5:])
                
                if row['id'] in results_best_model:
                    results_best_model[row['id']].append(se)
                else:
                    results_best_model[row['id']] = [se]
        

        # log scale
        # h = np.log(h)
        #plot results on plotly
        # if len(results_best_model) != len(h):
        #     st.write('Error in prediction')
        #     return
        # option to take log of Se
        # take_log = st.checkbox("Take log of Se", value=False)      
        
        for key, value in results_best_model.items():
            
            # get real values of key
            df_key = df[df['id'] == key]
            h_true = df_key['mbar'].values
            true_se = df_key['swc'].values

            fc_330_index = h.tolist().index(330)
            pwp_15000_index = h.tolist().index(15000)
            # if take_log:
            #     value = np.array(value)
            #     value = np.log(value)

            result_fc_330 = value[fc_330_index]
            result_pwp_15000 = value[pwp_15000_index]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h, y=value, mode='lines', name='Prediction'))
            # true values as dots
            fig.add_trace(go.Scatter(x=h_true, y=true_se, mode='markers', name='True', marker=dict(color='purple', size=5)))
            fig.add_trace(go.Scatter
            (x=[330], y=[result_fc_330], mode='markers', name='fc_330', marker=dict(color='red', size=10)))
            fig.add_trace(go.Scatter
            (x=[15000], y=[result_pwp_15000], mode='markers', name='pwp_15000', marker=dict(color='green', size=10)))
            # line for fc and pwp
            fig.add_shape(type="line", x0=330, y0=0, x1=330, y1=result_fc_330, line=dict(color="red", width=1))
            fig.add_shape(type="line", x0=15000, y0=0, x1=15000, y1=result_pwp_15000, line=dict(color="green", width=1))

            fig.update_layout(title=f'Se vs h for {key}', xaxis_title='h', yaxis_title='Se')
            st.plotly_chart(fig)




if __name__ == "__main__":
    run()


# Output -> + mbar_100, mbar_330, mbar_15000 -> 3 predictions for field capacity and permanent w point, available water = fc - pwp, available_water_100, available_water_330

# fc = 330, pwp = 15000

# column selection at the input time -> id=sample_id, swc = ...

# best model - first sample plot at the very top before tables

# first demo or upload