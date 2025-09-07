
import numpy as np
from scipy.stats import norm


def black_scholes(S, K, r, sigma, T, call):
    d0 = ( np.log(S/K) + (r + .5 * sigma ** 2) * T ) / ( sigma * np.sqrt(T) )
    d1 = d0 - sigma * np.sqrt(T)

    option_type = int(call == True) * 2 - 1

    option_price = (option_type * S * norm.cdf(option_type * d0) 
                    - option_type * K * np.exp(-r * T) * norm.cdf(option_type * d1))

    return option_price


class EuropeanOption():

    def __init__(self, S, K, r, sigma, T, call):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.call = call
        self.option_type = int(call == True) * 2 - 1

    def get_option_attributes(self):
        option_type_str = "call" if self.call else "put"
        print(f"European {option_type_str} option:")
        print(f"- Option price: {self.price:.4f}")
        print(f"- Underlying price: {self.S}")
        print(f"- Strike price: {self.K}")
        print(f"- Interest rate: {self.r*100:.2f}%")
        print(f"- Volatility: {self.sigma*100:.2f}%")
        print(f"- Time to expiry: {self.T}")

    @property
    def price(self):
        option_price = black_scholes(self.S, 
                                     self.K, 
                                     self.r, 
                                     self.sigma,
                                     self.T, 
                                     self.call)
        
        return option_price

    @property
    def d0(self):
        d0 = (( np.log(self.S / self.K) + (self.r + .5 * self.sigma ** 2) * self.T ) 
              / ( self.sigma * np.sqrt(self.T) ))
        return d0
    
    @property
    def d1(self):
        d1 = self.d0 - self.sigma * np.sqrt(self.T)
        return d1

    @property
    def delta(self):
        delta = self.option_type * norm.cdf(self.option_type * d0)

        return delta
    
    @property
    def gamma(self):
        if S is None:
            S = self.S

        d0 = self.d0(S)
        gamma = norm.pdf(d0) / (S * self.sigma * np.sqrt(self.T))

        return gamma
    
    @property
    def vega(self):
        if S is None:
            S = self.S

        d0 = self.d0(S)
        vega = S * norm.pdf(d0) * np.sqrt(T)

        return vega

