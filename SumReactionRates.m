function Gamma_list = SumReactionRates(T_photon, T_neutrino, n_baryon, Y_list, reactions, globals)

    Kelvin   = globals.Kelvin;
    kilogram = globals.kilogram;
    meter    = globals.meter;
    CL       = globals.CL;
    KB       = globals.KB;
    ME       = globals.ME;
    MU       = globals.MU;
    
    N_nuclei               = globals.N_nuclei;
    N_reactions            = globals.N_reactions;
    reaction_matrix        = reactions.reaction_matrix;
    log_forward_rates      = reactions.log_forward_rates;
    nuclear_inverse_params = reactions.nuclear_inverse_params;
    
    T_nine            = T_photon / (1E9 * Kelvin);
    z_photon          = ME*CL^2/(KB*T_photon);
    z_neutrino        = ME*CL^2/(KB*T_neutrino);
    rho_n             = n_baryon * MU;
    mass_density_unit = (0.001 * kilogram) / (0.01 * meter)^3; % g/cm^3
    
    %-------------------- input forward reaction rates -------------------------
    forward_rate_slice = zeros(N_reactions, 1);
    inverse_rate_slice = zeros(N_reactions, 1);
    
    [forward_rate_slice(1), inverse_rate_slice(1)] = WeakReactionRates(z_photon, z_neutrino, globals); % n -> p
    thermonuclear_rates = zeros(1, N_reactions);
    thermonuclear_rates(2:end) = exp(interp1(log_forward_rates(:,1), log_forward_rates(:,2:end), log(T_nine), 'spline', -Inf));
    if T_nine < 10 && T_nine > 0.001
        thermonuclear_rates(3)  = 66.2*(1+18.9*T_nine); % d + n > g + t, see NuclearForwardRates.dat
        thermonuclear_rates(7)  = 1.78E-9; % T > He3
        thermonuclear_rates(10) = 1.67E9*T_nine^(-2/3)*exp(-4.872*T_nine^(-1/3))*(1+0.086*T_nine^(1/3)-0.455*T_nine^(2/3)-0.272*T_nine+0.148*T_nine^(4/3)+0.225*T_nine^(5/3));
            % t + t > n + n + a
        thermonuclear_rates(11) = 6.62*(1+905*T_nine); % He3 + n > g + a
        thermonuclear_rates(14) = 2.21E5*T_nine^(-2/3)*exp(-7.720/T_nine^(1/3))*(1+2.68*T_nine^(2/3)+0.868*T_nine+0.192*T_nine^(4/3)+0.174*T_nine^(5/3)+0.044*T_nine^2);
            % He3 + t > g + Li6
        thermonuclear_rates(15) = 5.46E9*(T_nine/(1+0.128*T_nine))^(5/6)*T_nine^(-3/2)*exp(-7.733*(T_nine/(1+0.128*T_nine))^(-1/3)); % He3 + t > He4 + d
        thermonuclear_rates(16) = 7.71E9*(T_nine/(1+0.115*T_nine))^(5/6)*T_nine^(-3/2)*exp(-7.733*(T_nine/(1+0.115*T_nine))^(-1/3)); % He3 + t > n + p + a
        thermonuclear_rates(21) = 5.10E3; % Li6 + n > Li7 + g
        thermonuclear_rates(22) = 1.683E8*(1-0.261*(T_nine/(1+49.18*T_nine))^(3/2)*T_nine^(-3/2)) + 2.5432E9*T_nine^(-3/2)*exp(-2.39/T_nine); % Li6 + n > t + a
        thermonuclear_rates(25) = 1.48E12*T_nine^(-2/3)*exp(-10.135*T_nine^(-1/3)); % Li6 + d > Be7 + n
        thermonuclear_rates(26) = 1.48E12*T_nine^(-2/3)*exp(-10.135*T_nine^(-1/3)); % Li6 + d > Li7 + p
    end
    ratio_inverse_forward = nuclear_inverse_params(:,2) .* T_nine.^nuclear_inverse_params(:,3) .* exp(nuclear_inverse_params(:,4)./T_nine);
    
    for i = 2 : N_reactions
        [N_forward_sum, N_inverse_sum] = deal(0);
        for j = 1 : N_nuclei
            if reaction_matrix(i,j) < 0
                N_forward_sum = N_forward_sum - reaction_matrix(i,j);
            else
                N_inverse_sum = N_inverse_sum + reaction_matrix(i,j);
            end
        end
        forward_rate_slice(i) = (rho_n/mass_density_unit).^(N_forward_sum-1) .* thermonuclear_rates(i);
        inverse_rate_slice(i) = (rho_n/mass_density_unit).^(N_inverse_sum-N_forward_sum) .* ratio_inverse_forward(i) .* forward_rate_slice(i);
    end
    
    %-------------------- calculate Gamma_i ------------------------
    Gamma_list = zeros(1, N_nuclei);
    for i = 1 : N_reactions
        [YY_forward, YY_inverse] = deal(1);
        for j = 1 : N_nuclei
            if reaction_matrix(i,j) < 0
                YY_forward = YY_forward * Y_list(j).^abs(reaction_matrix(i,j)) ./ factorial(abs(reaction_matrix(i,j)));
            else
                YY_inverse = YY_inverse * Y_list(j).^reaction_matrix(i,j) ./ factorial(reaction_matrix(i,j)); % note that factorial(0)=1
            end
        end
        single_reaction_net = inverse_rate_slice(i).*YY_inverse - forward_rate_slice(i).*YY_forward;
        Gamma_list = Gamma_list - single_reaction_net.*reaction_matrix(i,:);
    end
    Gamma_list = Gamma_list';
end

function [weak_forward_rate, weak_inverse_rate] = WeakReactionRates(z_photon, z_neutrino, globals)

    second = globals.second;
    K_weak = 6.9503E-4 .* second.^(-1);
    
    weak_forward_rate = K_weak .* integral(@(x) WeakForwardIntegrand(x, z_photon, z_neutrino, globals), 1, Inf); % n -> p 
    weak_inverse_rate = K_weak .* integral(@(x) WeakInverseIntegrand(x, z_photon, z_neutrino, globals), 1, Inf); % p -> n 
end

function integrand_total = WeakForwardIntegrand(x, z_photon, z_neutrino, globals)

    ME = globals.ME;
    MN = globals.MN;
    MP = globals.MP;
    q  = (MN-MP)/ME;
    
    integrand_1 = x.*(x+q).^2.*sqrt(x.^2-1) ./ ( (1+exp(+x.*z_photon)) .* (1+exp(-(x+q).*z_neutrino)) );
    integrand_2 = x.*(x-q).^2.*sqrt(x.^2-1) ./ ( (1+exp(-x.*z_photon)) .* (1+exp(+(x-q).*z_neutrino)) );
    integrand_total = integrand_1 + integrand_2;
end

function integrand_total = WeakInverseIntegrand(x, z_photon, z_neutrino, globals)

    ME = globals.ME;
    MN = globals.MN;
    MP = globals.MP;
    q  = (MN-MP)/ME;
    
    integrand_1 = x.*(x-q).^2.*sqrt(x.^2-1) ./ ( (1+exp(+x.*z_photon)) .* (1+exp(-(x-q).*z_neutrino)) );
    integrand_2 = x.*(x+q).^2.*sqrt(x.^2-1) ./ ( (1+exp(-x.*z_photon)) .* (1+exp(+(x+q).*z_neutrino)) );
    integrand_total = integrand_1 + integrand_2;
end
