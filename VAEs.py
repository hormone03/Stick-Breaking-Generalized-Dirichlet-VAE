import torch
from torch.nn import functional as F
from utils import parser, init_weight_mean_var, uniform_low, uniform_high, input_ndims, \
    logistic_func, beta_func, prior_beta, prior_alpha , glogit_prior_mu, prior_sigma
from torch.distributions.gamma import Gamma
from encoderDecoder import stickBrEncoder, stickBrDecoder
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = not args.no_cuda and torch.cuda.is_available()


class VAE(object):
    def __init__(self, target_distribution, latent_distribution, prior_param1, prior_param2):
        self.target_distribution = target_distribution
        self.latent_distribution = latent_distribution
        self.prior_param1 = prior_param1.to(device)
        self.prior_param2 = prior_param2.to(device)

        #self.init_weights(self.encoder_layers)
        #self.init_weights(self.decoder_layers)
        #Generates uniformly distributed random sample from the half-open interval [low, high).
        self.uniform_distribution = torch.distributions.uniform.Uniform(low=uniform_low, high=uniform_high)

    def init_weights(self, layers):
        for layer in layers:
            with torch.no_grad():
                layer_size = (layer.out_features, layer.in_features)
                # init_weight_mean_var = (0, .001)
                layer.weight = torch.nn.Parameter(torch.normal(*init_weight_mean_var, layer_size))
                layer.bias = torch.nn.Parameter(torch.zeros(layer.out_features))

    def ELBO_loss(self, recon_x, x, param1, param2, kl_divergence, parametrization=None):
        n_samples = len(recon_x)
        #x = x.view(-1, input_ndims)
        #.all() returns true if all the elements are true. 
        # torch.isfinite() returns true for each real value, and false for NAN
        #if not torch.isfinite(recon_x.log()).all():
            #raise AssertionError('Reconstructed x.log not finite!: ', recon_x.log())
        #reconstruction_loss = - (x * recon_x.log() + (1 - x))
        reconstruction_loss = F.binary_cross_entropy(recon_x.view(-1, input_ndims), x.view(-1, input_ndims), reduction='sum')
        if parametrization == 'GDVAE':
            regularization_loss = kl_divergence(param1, param2)
        elif parametrization == 'GDWO':
            regularization_loss = kl_divergence(param1, param2)
        elif parametrization == 'Dirichlet_dist':
            regularization_loss = kl_divergence(param1, param2)
        elif parametrization == 'Gaussian':
            regularization_loss = kl_divergence(param1, param2)
        elif parametrization == 'Gauss_Logit':
            regularization_loss = kl_divergence(param1, param2)
        else:
            regularization_loss = torch.stack([kl_divergence(param1[i], param2[i]) for i in range(n_samples)])
            
        
        return reconstruction_loss,  regularization_loss

    def reparametrize(self, param1, param2, parametrization=None):
        if parametrization == 'Gaussian':
            param1 = torch.max(torch.tensor(0.0001), param1).to(device)
            param2 = torch.max(torch.tensor(0.0001), param2).to(device)
            # .new() constructs new data of the same tensor type
            epsilon = param2.data.new(param2.size()).normal_()
            out = param1 + param2 * epsilon

        elif parametrization == 'Kumaraswamy':
            
            param1 = torch.max(torch.tensor(0.0001), param1).to(device)
            param2 = torch.max(torch.tensor(0.0001), param2).to(device)
            doc_vec = self.get_kumaraswamy_samples(param1, param2)
            z1 = torch.div(doc_vec,torch.reshape(torch.sum(doc_vec,1), (-1, 1)))
            # we sampled from the region of highest desnsity because sampling from z1 was noisy
            out = 1 - self.compute_stick_segment(z1)

        elif parametrization == 'GEM':
            param1 = self.softplus (param1).to(device)
            param2 = self.softplus (param2).to(device)
            # for GEM, param1 == alpha, param2 == beta
            param1 = torch.clamp(param1, min=torch.tensor(0.0001).to(device), max=torch.tensor(1.0).to(device))
            param2 = torch.clamp(param2, min=torch.tensor(0.0001).to(device), max=torch.tensor(1.0).to(device))
            
            doc_vec = self.get_GEM_samples(param1, param2)
            z1 = torch.div(doc_vec,torch.reshape(torch.sum(doc_vec,1), (-1, 1)))
            out = self.compute_stick_segment(z1)

        elif parametrization == 'Gauss_Logit':
           
            param1 = torch.max(torch.tensor(0.0001), param1).to(device)
            param2 = torch.max(torch.tensor(0.0001), param2).to(device)
            epsilon = param2.data.new(param2.size()).normal_()
            z = param1 + param2 * epsilon
            #z1 = torch.div(z,torch.reshape(torch.sum(z,1), (-1, 1)))
            out = self.compute_stick_segment(z)
            
        elif parametrization == 'Dirichlet_dist':
            mu = torch.max(torch.tensor(0.0001), param1).to(device)
            logvar = torch.max(torch.tensor(0.0001), param2).to(device)
            
            gam1 = torch.squeeze(self.my_random_gamma(shape = (1,),alpha=mu+ torch.tensor(self.B)))
            
            eps = (self.calc_epsilon(gam1,mu+torch.tensor(self.B))).detach()
            #uniform variables for shape augmentation of gamma
            u = torch.rand(1,mu.shape[0],args.category)
           
            doc_vec = self.gamma_h_boosted(eps,u,mu,u.shape[0])
            
            z1 = torch.div(doc_vec,torch.reshape(torch.sum(doc_vec,1), (-1, 1)))
            z = self.softplus(z1)
            out = self.compute_stick_segment(z)
            #out = elf.softplus(out) #### decoupling smoothness and sparsity
                    
        elif parametrization == 'GDVAE': 
            mu = torch.max(torch.tensor(0.0001), param1).to(device)
            logvar = torch.max(torch.tensor(0.0001), param2).to(device)
            
            gam1 = torch.squeeze(self.my_random_gamma(shape = (1,),alpha=mu+ torch.tensor(self.B)))
            gam2 = torch.squeeze(self.my_random_gamma(shape = (1,),alpha=logvar+ torch.tensor(self.B)))
            
            eps = (self.calc_epsilon(gam1,mu+torch.tensor(self.B))).detach()
            eps2 = (self.calc_epsilon(gam2,logvar+torch.tensor(self.B))).detach()
            #uniform variables for shape augmentation of gamma
            u = torch.rand(1,mu.shape[0],args.category)
            #u=torch.FloatTensor(self.n_sample, self.B).uniform_(1, self.n_topic)
            #this is the sampled gamma for this document, boosted to reduce the variance of the gradient
            doc_vec1 = self.gamma_h_boosted(eps,u,mu,u.shape[0])
            doc_vec2 = self.gamma_h_boosted(eps2,u,logvar,u.shape[0])
            
            z1 = torch.div(doc_vec1,torch.reshape(torch.sum(doc_vec1,1), (-1, 1)))
            z2 = torch.div(doc_vec2,torch.reshape(torch.sum(doc_vec2,1), (-1, 1)))
            pi_1 = self.compute_stick_segment(z1)
            pi_2 = self.compute_stick_segment(z2)
            out = pi_1 + (1- pi_2)
            
        elif parametrization == 'GDWO': 
            mu = torch.max(torch.tensor(0.0001), param1).to(device)
            logvar = torch.max(torch.tensor(0.0001), param2).to(device)
            
            doc_vec1 = torch.squeeze(self.my_random_gamma(shape = (1,),alpha=mu+ torch.tensor(self.B)))
            doc_vec2 = torch.squeeze(self.my_random_gamma(shape = (1,),alpha=logvar+ torch.tensor(self.B)))            
            z1 = torch.div(doc_vec1,torch.reshape(torch.sum(doc_vec1,1), (-1, 1)))
            z2 = torch.div(doc_vec2,torch.reshape(torch.sum(doc_vec2,1), (-1, 1)))
            pi_1 = self.compute_stick_segment(z1)
            pi_2 = self.compute_stick_segment(z2)
            out = pi_1 + (1- pi_2)
            
        else:
            print("############# Parameterization is not defined ##########################")
            
        return out
    
    def gamma_h_boosted(self, epsilon, u, alpha,model_B):
        #(eps,u,mu,batch_size)
        epsilon = epsilon.to(device)
        u = u.to(device)
        alpha = alpha.to(device)
        #Me: this calculate z inside section 3.4, z_tilder there now implies EQ2
        #Me: Note that all the lines till u_power generate u
        """
        Reparameterization for gamma rejection sampler with shape augmentation.
        """
        #B = u.shape.dims[0] #u has shape of alpha plus one dimension for B
        #Me: Note that alpha shape is not noumber of topic
        B = u.shape[0]
        K = alpha.shape[1]#(batch_size,K)
        r = torch.arange(0,B) #Me: range(0, n), n=B here 
        #Me: reshape r
        rm = (torch.reshape(r,[-1,1,1])).type(torch.FloatTensor).to(device) #dim Bx1x1
        #Me: tile expand and Copy a Tensor: tf.tile(input,multiples,name=None)
        #Me: xy = tf.tile(xs, multiples = [2, 3]) means repeat generate twice in x-ddirection, 3 times in y-direction
        #Me:https://www.tutorialexample.com/understand-tensorflow-tf-tile-expand-a-tensor-tensorflow-tutorial/
        alpha_vec = torch.reshape(torch.tile(alpha,(B,1)),(model_B,-1,K)) + rm #dim BxBSxK + dim Bx1
        alpha_vec = alpha_vec.to(device)
        u_pow = torch.pow(u,1./alpha_vec)+1e-10
        gammah = self.gamma_h(epsilon, alpha + torch.tensor(B))
        return torch.prod(u_pow,axis=0)*gammah

    
    def calc_epsilon(self,gamma,alpha):
        return torch.sqrt(9.*alpha-3.)*(torch.pow(gamma/(alpha-1./3.),1./3.)-1.)

    def my_random_gamma(self,shape, alpha, beta=1.0):

        alpha = torch.ones(shape).to(device) * alpha
        beta = torch.ones(shape).to(device) * torch.tensor(beta).to(device)
        
        gamma_distribution = Gamma(alpha, beta)
    
        return gamma_distribution.sample()
    
    def gamma_h(self, epsilon, alpha):
        #Me: gamma_h is from equ_2: z=h_gamma(eps, alpha)= EQ2
        """
        Reparameterization for gamma rejection sampler without shape augmentation.
        """
        b = alpha - 1./3.
        c = 1./torch.sqrt(9.*b)
        v = 1.+epsilon*c
        
        return b*(v**3) 

    
    
    def compute_stick_segment(self, v):
        #n_samples = v.size()[0]
        n_dims = v.size()[1]
        pi = torch.ones(size=v.size()).to(device)
        for idx in range(n_dims):
            product = 1
            for sub_idx in range(idx):
                product *=1-v[:,sub_idx]
            pi[:,idx] = v[:,idx] * product
        return pi


    def get_kumaraswamy_samples(self, param1, param2):
        # u is analogous to epsilon noise term in the Gaussian VAE
        u_hat = self.uniform_distribution.sample([1]).squeeze().to(device)
        #Equation 6 in their paper
        v = (1 - u_hat.pow(1 / param2.to(device))).pow(1 / param1.to(device))
        #v = (u_hat * param1 * torch.lgamma(param1).exp()).pow(1 / param1) / param2
        return v.to(device)  # sampled fractions

    def get_GEM_samples(self, param1, param2):
        #param1 = torch.clamp(param1, min=torch.tensor(0.0001).to(device), max=torch.tensor(1.0).to(device))
        #param2 = torch.clamp(param2, min=torch.tensor(0.0001).to(device), max=torch.tensor(1.0).to(device))
        u_hat = self.uniform_distribution.sample([1]).squeeze()
        #Equation 4 in thier paper
        v = (u_hat * param1 * torch.lgamma(param1).exp()).pow(1 / param1) / param2

        #poor_approx_idx = torch.where((param1 >= 1) * (1 - .94*u_hat*param1.log() >= -.42)) # Kingma & Welling (2014)
        #poor_approx_idx = torch.where(param1 >= 1)
        #if poor_approx_idx[0].nelement() != 0:
            #v1 = param1 / (param1 + param2)
            #v[poor_approx_idx] = v1[poor_approx_idx]

        return v.to(device)

class StickBreakingVAE(torch.nn.Module, stickBrEncoder, stickBrDecoder, VAE):
    def __init__(self, parametrization):
        super(StickBreakingVAE, self).__init__()
        stickBrEncoder.__init__(self)
        stickBrDecoder.__init__(self)
        self.parametrization = parametrization
        self.B = 1.0

        if parametrization == 'Kumaraswamy':
            VAE.__init__(self, target_distribution=torch.distributions.beta.Beta,
                         latent_distribution=torch.distributions.kumaraswamy.Kumaraswamy,
                         prior_param1= prior_alpha,
                         prior_param2= prior_beta)
        elif parametrization == 'Gauss_Logit':
            VAE.__init__(self, target_distribution=None,
                         latent_distribution=None,
                         prior_param1= glogit_prior_mu,
                         prior_param2= prior_sigma ** 2)
        elif parametrization == 'Gaussian':
            VAE.__init__(self, target_distribution=None,
                         latent_distribution=None,
                         prior_param1= glogit_prior_mu,
                         prior_param2= prior_sigma ** 2)
        elif parametrization == 'GEM':
            # Gamma distribution used to approximate beta distribution
            VAE.__init__(self, target_distribution=torch.distributions.gamma.Gamma,
                         latent_distribution=torch.distributions.gamma.Gamma,
                         prior_param1= prior_alpha,
                         prior_param2= prior_beta)

        elif parametrization == 'Dirichlet_dist':
            VAE.__init__(self, target_distribution=None,
                         latent_distribution=None,
                         prior_param1= prior_alpha,
                         prior_param2= prior_beta)
        elif parametrization == 'GDVAE':
            VAE.__init__(self, target_distribution=None,
                         latent_distribution=None,
                         prior_param1= prior_alpha,
                         prior_param2= prior_beta)
        elif parametrization == 'GDWO':
            VAE.__init__(self, target_distribution=None,
                         latent_distribution=None,
                         prior_param1= prior_alpha,
                         prior_param2= prior_beta)



    def forward(self, x):
        param1, param2 = self.encode(x)
        if self.training:
            pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
        else:
            # reconstruct random samples from the area of highest density
            if self.parametrization == 'Kumaraswamy':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
                #highest_density = (1 - self.latent_distribution(param1, param2).mean.pow(1 / param2)).pow(1 / param1)
                #pi = self.compute_stick_segment(highest_density)
            elif self.parametrization == 'GEM':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
                
            elif self.parametrization == 'GEM_':
                #pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
                param1 = torch.clamp(param1, min=torch.tensor(0.0001).to(device), max=torch.tensor(1.0).to(device))
                param2 = torch.clamp(param2, min=torch.tensor(0.0001).to(device), max=torch.tensor(1.0).to(device))
                highest_density = torch.lgamma(param1).exp().mul(param1).\
                    mul(torch.distributions.Beta(param1, param2).mean).pow(1 / param1).div(param2)
                z1 = torch.div(highest_density,torch.reshape(torch.sum(highest_density,1), (-1, 1)))
                pi = self.compute_stick_segment(z1)
            elif self.parametrization == 'Gaussian':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
            elif self.parametrization == 'Gauss_Logit':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
            elif self.parametrization == 'Dirichlet_dist':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
            elif self.parametrization == 'GDVAE':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
            elif self.parametrization == 'GDWO':
                pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
               
               
        reconstructed_x = self.decode(pi)

        #if self.parametrization == 'Gauss_Logit':
            #param2 = torch.stack([torch.diag(param2[i].pow(2)) for i in range(len(param2))])
            
        
            
        self.outLength = len(reconstructed_x)

        return reconstructed_x, param1, param2

    def kl_divergence(self, param1, param2):\
    
        kl_switcher = dict(Kumaraswamy=self.kumaraswamy_kl_divergence,
                           GEM=self.gamma_kl_divergence,
                           Gauss_Logit=self.gauss_logit_kl_divergence,
                           Gaussian=self.gaussian_kl_divergence,
                           Dirichlet_dist=self.dir_kl_divergence,
                           GDVAE=self.GD_kl_divergence,
                           GDWO=self.GD_kl_divergence)
        kl_divergence_func = kl_switcher.get(self.parametrization)

        #assert((param1 != 0).all(), f'Zero at alpha indices: {torch.nonzero((param1!=0) == False, as_tuple=False).squeeze()}')
        #assert((param2 != 0).all(), f'Zero at beta indices: {torch.nonzero((param2!=0) == False, as_tuple=False).squeeze()}')

        return kl_divergence_func(param1, param2)

    def gauss_logit_kl_divergence(self, mu, sigma):
        mu = torch.max(torch.tensor(0.0001), mu).to(device)
        sigma = torch.max(torch.tensor(0.0001), sigma).to(device)
        
        prior_mean   = self.prior_param1.expand_as(mu)
        prior_var    = self.prior_param2.expand_as(sigma)
        prior_logvar = self.prior_param2.log().expand_as(sigma)
        var_division    = sigma  / prior_var
        diff            = mu - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - sigma
        # put KLD together
        kl = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - args.category )
        return kl.sum()
    
    def gaussian_kl_divergence(self, mu, sigma):
        mu = torch.max(torch.tensor(0.0001), mu).to(device)
        sigma = torch.max(torch.tensor(0.0001), sigma).to(device)
        
        prior_mean   = self.prior_param1.expand_as(mu)
        prior_var    = self.prior_param2.expand_as(sigma)
        prior_logvar = self.prior_param2.log().expand_as(sigma)
        var_division    = sigma  / prior_var
        diff            = mu - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - sigma
        # put KLD together
        kl = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - args.category )
        return kl.sum()
    
    
    def _gamma_kl_divergence(self, alpha, beta):
        #alpha = torch.max(torch.tensor(0.0001), alpha).to(device)
        alpha = torch.clamp(alpha, min=torch.tensor(0.6).to(device), max=torch.tensor(0.0001).to(device))
        beta = torch.max(torch.tensor(0.0001), beta).to(device)
        prior_param1 = self.prior_param1.expand_as(alpha)
        prior_param2 = self.prior_param2.expand_as(beta)
        
        kl = prior_param1*torch.log(beta) - prior_param1*torch.log(prior_param2) - torch.lgamma(alpha)
        kl += torch.lgamma(prior_param1)  + (alpha - prior_param1)*torch.digamma(alpha)
        kl += -(beta - prior_param2)*torch.div(alpha, beta)
        return kl.sum() #(kl1 + kl2).sum()
    
    def gamma_kl_divergence(self, alpha, beta):
        alpha = torch.clamp(alpha, min=torch.tensor(0.6).to(device), max=torch.tensor(0.0001).to(device))
        #alpha = torch.max(torch.tensor(0.0001), alpha).to(device)
        beta = torch.max(torch.tensor(0.0001), beta).to(device)
        #beta = torch.clamp(beta, min=torch.tensor(0.9).to(device), max=torch.tensor(0.0001).to(device))
        alpha = self.softplus (alpha).to(device)
        beta = self.softplus (beta).to(device)
        q1 = self.latent_distribution(alpha, 1)
        q2 = self.latent_distribution(beta, 1)
        p1 = self.target_distribution(self.prior_param1, 1)
        p2 = self.target_distribution(self.prior_param2, 1)
        kl1 = torch.distributions.kl_divergence(q1, p1)
        kl2 = torch.distributions.kl_divergence(q2, p2)
        return (kl1 + kl2).sum()
   
   
    def kumaraswamy_kl_divergence(self, alpha, beta):
        alpha = torch.clamp(alpha, min=torch.tensor(0.6).to(device), max=torch.tensor(0.0001).to(device))
        #alpha = torch.max(torch.tensor(0.0001), alpha).to(device)
        #beta = torch.max(torch.tensor(0.0001), beta).to(device)
        beta = torch.clamp(beta, min=torch.tensor(0.6).to(device), max=torch.tensor(0.0001).to(device))
        alpha = self.softplus (alpha).to(device)
        beta = self.softplus (beta).to(device)
        #psi_b_taylor_approx = torch.lgamma(beta) 
        psi_b_taylor_approx = beta.log() - 1. / beta.mul(2) - 1. / beta.pow(2).mul(12)

        kl = ((alpha - self.prior_param1) / alpha) * (-0.57721 - psi_b_taylor_approx - 1 / beta)
        kl += (alpha * beta).log() + beta_func(self.prior_param1, self.prior_param2).log()  # normalization constants
        kl += - (beta - 1) / beta
        #kl += (self.prior_param2 - 1) * beta * (1/(1 + alpha*beta)* beta_func(1/(alpha + torch.tensor(1)), beta)).sum() # we take a finite approximation of the infinite sum
        kl += torch.stack([1. / (i + alpha * beta) * beta_func(i / (alpha + torch.tensor(1)), beta) for i in range(1, 11)]).sum(axis=0) \
              * (self.prior_param2 - 1) * beta  # 10th-order Taylor approximation
        #print("===>" * 4)
        #print(kl.sum().item())
        #print("===>" * 4)
        return kl
    
   
    def GD_kl_divergence(self, encoderAlpha, encoderBeta):
        encoderAlpha = torch.max(torch.tensor(0.0001), encoderAlpha).to(device)
        encoderBeta = torch.max(torch.tensor(0.0001), encoderBeta).to(device)
        prior_alpha = self.prior_param1.expand_as(encoderAlpha)
        prior_beta = self.prior_param2.expand_as(encoderBeta)
        
        alphaDiff = encoderAlpha - prior_alpha
        betaDiff = encoderBeta - prior_beta 
        priorsParamsSum = prior_alpha + prior_beta
        encodeParamsSum = encoderAlpha + encoderBeta
        
        numerator = torch.lgamma(priorsParamsSum) + torch.lgamma(prior_alpha) + torch.lgamma(prior_beta)
        denomerator = torch.lgamma(encodeParamsSum) + torch.lgamma(encoderAlpha) + torch.lgamma(encoderBeta)
        firstTerm = numerator - denomerator
        secondTerm = alphaDiff * (torch.digamma(encoderAlpha) - torch.digamma(priorsParamsSum))
        thirdTerm = betaDiff * (torch.digamma(encoderBeta) - torch.digamma(priorsParamsSum))
        analytical_kld = torch.sum((firstTerm + secondTerm + thirdTerm), dim=1)
        return analytical_kld
    
    def dir_kl_divergence(self, mu, sigma):
        alpha = torch.max(torch.tensor(0.0001), mu).to(device)
        prior_alpha = self.prior_param1.expand_as(mu).to(device)
        
        analytical_kld = torch.lgamma(torch.sum(alpha,dim=1))-torch.lgamma(torch.sum(prior_alpha,dim=1))
        analytical_kld-=torch.sum(torch.lgamma(alpha),dim=1)
        analytical_kld+=torch.sum(torch.lgamma(prior_alpha),dim=1)
        minus = alpha-prior_alpha
        lastExpression = torch.sum(torch.multiply(minus,torch.digamma(alpha)-torch.reshape(torch.digamma(torch.sum(alpha,1)),(-1,1))),1)
        analytical_kld+=lastExpression
        return analytical_kld