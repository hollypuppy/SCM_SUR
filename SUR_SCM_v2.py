import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, random, numpy as jnp
import numpyro as npr
from numpyro import infer, distributions as dist


####################
# helper functions #
####################

@jit
def make_beta(beta_off: jnp.ndarray, 
              lambd: jnp.ndarray,
              eta: jnp.ndarray, 
              tau: jnp.ndarray):
    
    cache = jnp.tan(0.5 * jnp.pi * lambd) * jnp.tan(0.5 * jnp.pi * eta)
    tau_ = jnp.tan(0.5 * jnp.pi * tau)
    out = jnp.diag(cache) @ (beta_off * tau_)

    return out


@jit
def make_Sigma(Sigma_diag):
    L = Sigma_diag.shape[1]
    Sigma = Sigma_diag[:,:,jnp.newaxis] * jnp.eye(L)

    return Sigma


@jit
def make_Phi(Sigma: jnp.ndarray, 
             F: jnp.ndarray, 
             beta: jnp.ndarray):
    # Sigma: KxLxL; f: TxL; beta: LxJ; Phi: TxJxK
    Phi = F[jnp.newaxis, :, :] @ Sigma @ beta[jnp.newaxis, :, :] # 1xTxL @ KxLxL @ 1xLxJ = KxTxJ
    
    return Phi


def make_Y(Y_1_pre: jnp.ndarray, 
           Y_1_post: jnp.ndarray, 
           Y_0: jnp.ndarray):
    # Y_0: (J-1)xTxK; Y_1_pre: T_prexK; Y_1_post: T_postxK
    assert Y_0.shape[1] == Y_1_pre.shape[0] + Y_1_post.shape[0]
    # This line ensures that the number of columns in Y_0 is equal to the sum of the number of rows in Y_1_pre and Y_1_post.
    # It's a check to make sure the concatenation and stacking are valid.
    Y = jnp.concatenate([Y_1_pre, Y_1_post]) #TxK
    Y_reshaped = Y.reshape(1, *Y.shape) #1xTxK
    Y = jnp.concatenate([Y_reshaped, Y_0], axis=0) #JxTxK
    Y = jnp.transpose(Y, axes=(1, 0, 2)) #TxJxK
    return Y


#########
# model #
#########        
        
def model_sur_scm(Y_0: jnp.ndarray, Y_1_pre: jnp.ndarray, L: int):
    # initialize
    J = Y_0.shape[0] + 1
    T = Y_0.shape[1]
    K = Y_0.shape[2]
    T_post = T - Y_1_pre.shape[0]

    # define params and priors
    eta = npr.sample("eta", dist.Uniform())
    
    with npr.plate("L",L):
        lambd = npr.sample("lambda", dist.Uniform())
        
    with npr.plate("J", J):
        tau = npr.sample("tau", dist.Uniform())
        
    with npr.plate("L", L, dim=-2), npr.plate("J", J, dim=-1):
        beta_off = npr.sample("beta_off", dist.Normal())
    
    with npr.plate("K", K, dim=-2), npr.plate("L", L, dim=-1):
        Sigma_diag = npr.sample("Sigma_diag", dist.Normal())
        # Sigma = npr.sample("Sigma", dist.Normal().expand([L]).to_event(1))

    with npr.plate("T", T, dim=-2), npr.plate("L", L, dim=-1):
        F = npr.sample("f", dist.Normal())
        
    with npr.plate("K", K, dim=-2), npr.plate("J", J, dim=-1):
        kappa = npr.sample("kappa", dist.Normal())
        
    with npr.plate("K", K, dim=-2), npr.plate("T", T, dim=-1):
        delta = npr.sample("delta", dist.Normal(scale=2))

    with npr.plate("K", K):
        sig_err = npr.sample('L_sigma', dist.HalfCauchy())
    
    
    # data augmentation
    with npr.plate("T_post", T_post, dim=-2), npr.plate("K", K, dim=-1):
        Y_1_post = npr.sample("Y_1_post", dist.Normal().mask(False))
        # mask(False) indicates that no observations (masking) are associated with this variable

    # transform variables
    beta = make_beta(beta_off, lambd, eta, tau)
    Sigma = make_Sigma(Sigma_diag)
    Phi = make_Phi(Sigma, F, beta) # Phi: KxTxJ
    Y = make_Y(Y_1_pre, Y_1_post, Y_0) # Y: TxJxK
    
    # combine and reshape
    mu = Phi + delta[:, :, jnp.newaxis] + kappa[:, jnp.newaxis, :]
    mu_reshaped = mu.transpose([1, 2, 0]).reshape(T * J, K)
    Y_reshaped = Y.reshape(T * J, K)
    
    # likelihood
    with npr.plate("T*J", T * J, dim=-2), npr.plate("K", K, dim=-1):
        npr.sample("Y", dist.Normal(loc=mu_reshaped, scale=sig_err), obs=Y_reshaped)


########
# data #
########

def get_data(path: str = None):
    isGerman = path is None
    if isGerman:
        path = "german_unification.csv"

    dt = pd.read_csv(path, index_col=0)
    x_values =dt.columns.tolist()

    log_dt = dt.apply(np.log)
    inverse_dt = dt.apply(lambda x: 1/x)

    if (isGerman):
        T_pre = dt.columns.get_indexer(["1990"])[0]
        Y1_1_obs = jnp.array(dt.loc["West Germany"])
        Y2_1_obs = jnp.array(log_dt.loc["West Germany"])
        Y3_1_obs = jnp.array(inverse_dt.loc["West Germany"])
        Y_1_obs = jnp.column_stack((Y1_1_obs, Y2_1_obs,Y3_1_obs))

        # whitening
        normalize = lambda x: (x - x[T_pre]) / x[:T_pre + 1].std()


        def normal(df):
            whitening = {}
            for i in jnp.arange(df.shape[0]):
                i = int(i)
                whitening[df.iloc[i].name] = {'mu': df.iloc[int(i), T_pre], 'std': df.iloc[int(i), :T_pre].std()}
                df.iloc[i] = normalize(df.iloc[i])
            return df, whitening

        dt,whitening_1 = normal(dt)
        log_dt,whitening_2 = normal(log_dt)
        inverse_dt, whitening_3 = normal(inverse_dt)

        # parse
        Y1_1 = jnp.array(dt.loc["West Germany"])
        Y2_1 = jnp.array(log_dt.loc["West Germany"])
        Y3_1 = jnp.array(inverse_dt.loc["West Germany"])
        Y_1 = jnp.column_stack((Y1_1, Y2_1,Y3_1))
        Y_1_pre = Y_1[:T_pre]

        Y1_0 = jnp.array(dt.drop(["West Germany"], axis=0) ) #JxT
        Y2_0 = jnp.array(log_dt.drop(["West Germany"], axis=0)) #JxT
        Y3_0 = jnp.array(inverse_dt.drop(["West Germany"], axis=0))  # JxT
        Y_0 = jnp.stack((Y1_0, Y2_0, Y3_0), axis=-1)  #JxTxK
        Y_1_post_obs = Y_1[T_pre:]


    return x_values, Y_0, Y_1_obs, Y_1_pre, whitening_1, whitening_2, whitening_3


#######
# run #
#######

def main(args):
    # initializations
    rng_key = random.PRNGKey(args.seed)
    rng_key, rng_key_mcmc, rng_key_predict = random.split(rng_key, 3)
    x_values,Y_0, Y_1_obs, Y_1_pre, whitening1, whitening2, whitening3 = get_data()

    T = Y_0.shape[1]
    L = args.num_latent
    J = Y_0.shape[0] + 1
    # inference
    nuts_kernel = infer.NUTS(model_sur_scm, max_tree_depth=8, target_accept_prob=0.8)
    mcmc = infer.MCMC(nuts_kernel, num_warmup=args.iter, num_samples=args.iter, num_chains=1)
    mcmc.run(rng_key_mcmc,Y_0, Y_1_pre, L)

    # print
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()

    # posterior predictive
    ppd = infer.Predictive(model_sur_scm, posterior_samples, num_samples=args.iter, parallel=True) # Y：TxJxK
    Y_counterfactual = ppd(rng_key_predict,Y_0, Y_1_pre, L)["Y"]
    K=3
    Y_1_counterfactual = jnp.array(Y_counterfactual).reshape([args.iter,T,J,K])[:,:,0,:]
    Y_1_counterfactual = Y_1_counterfactual.reshape([args.iter,T,K])
    Y1_1_counterfactual = Y_1_counterfactual[:,:,0]
    Y2_1_counterfactual = Y_1_counterfactual[:,:,1]
    Y3_1_counterfactual = Y_1_counterfactual[:, :, 2]
    Y1_1_counterfactual *= whitening1['West Germany']['std']
    Y1_1_counterfactual += whitening1['West Germany']['mu']
    Y2_1_counterfactual *= whitening2['West Germany']['std']
    Y2_1_counterfactual += whitening2['West Germany']['mu']
    Y3_1_counterfactual *= whitening3['West Germany']['std']
    Y3_1_counterfactual += whitening3['West Germany']['mu']
    y1p_mu = Y1_1_counterfactual.mean(axis=0)
    y2p_mu = Y2_1_counterfactual.mean(axis=0)
    y3p_mu = Y3_1_counterfactual.mean(axis=0)




    #plot
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Serif']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot on the first subplot

    ax1.plot(x_values,Y_1_obs[:,0]) #Y_treated observations
    ax1.plot(x_values, y1p_mu) #Y_treated_conterfactual
    ax1.xaxis.set_major_locator(plt.MaxNLocator(10))

    ax1.set_title("GDP")

    # Plot on the second subplot
    ax2.plot(x_values,Y_1_obs[:, 1])
    ax2.plot(x_values, y2p_mu)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(10))

    ax2.set_title("log(GDP)")

    # Plot on the third subplot
    ax3.plot(x_values,Y_1_obs[:, 2])
    ax3.plot(x_values, y3p_mu)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    #ax3.axvline(x=1990, color='r', linestyle='--')
    ax3.set_title("1/GDP")
    plt.savefig('SUR_SCM_3outcomes.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-n", "--num-latent", default=10, type=int)
    parser.add_argument("-seed", default=20240208, type=int)
    parser.add_argument("-iter", default=2000, type=int)
    
    args = parser.parse_args()
    main(args)