clear
clc

tic
which_model          = 3; % choose the model: 1 (standard), 2 (exponential), 3 (stepwise)
[globals, reactions] = InputPhysicalData(which_model);
Kelvin               = globals.Kelvin;
N_nuclei             = globals.N_nuclei;
N_eta                = 32; % 32 is enough!
eta_list             = 11/4 * logspace(-10, -9, N_eta);
Y_eta                = zeros(N_eta, N_nuclei+which_model^2-2*which_model+4); % sliced variable, save the final BBN result and model informations, to plot the Y-eta figure
i_Planck             = round(1+(N_eta-1)*(10+log10(6.1374E-10)));
eta_list(i_Planck)   = 11/4 * 6.1374E-10;

parpool('local',8); % CPU workers
start_time        = datestr(now, 'dd-mmm-yyyy HH:MM');
data_queue        = parallel.pool.DataQueue;
wait_bar          = waitbar(0, sprintf('start at %s, working on the first loop', start_time), 'Name', 'BBNLab'); % show progress
wait_bar.UserData = [0 N_eta];
afterEach(data_queue, @(~, ~) UpdateWaitBar(wait_bar, start_time));
parfor i = 1 : N_eta
    T_photon_ini = 1E12 * Kelvin;
    eta_ini      = eta_list(i);
    
    % setting the model parameters and initial conditions
    scalar_params = zeros(1, which_model^2-2*which_model+2); % length is the number of model parameters, {1,2,3}->{1,2,5}
    vars_ini      = zeros(N_nuclei+7, 1);
    if which_model == 1
        scalar_params = NaN;
        [vars_ini, ~] = InitialConditions(T_photon_ini, eta_ini, scalar_params, globals);
    end
    if which_model == 2
        scalar_params             = [10 0]; % [lambda V_null]
        [vars_ini, V_null_update] = InitialConditions(T_photon_ini, eta_ini, scalar_params, globals);
        scalar_params(2)          = V_null_update; % update V_null
    end
    if which_model == 3
        scalar_params             = [10 0.01 0.24 0 70.2]; % [lambda_1 lambda_2 alpha V_null N_ini]
        [vars_ini, V_null_update] = InitialConditions(T_photon_ini, eta_ini, scalar_params, globals);
        scalar_params(4)          = V_null_update; % update V_null
    end
    
    % solve the BBN differential equations
    N_span = [0 10];
    options = odeset();
    if which_model == 1 || which_model == 2
        options = odeset('RelTol', 1E-10, 'AbsTol', 1E-10, 'MaxStep', 1E-3);
    end
    if which_model == 3
        options = odeset('RelTol', 1E-10, 'AbsTol', 1E-10, 'MaxStep', 2E-4);
    end
    [N_sol, vars_sol] = ode23s(@(N, vars) EvolutionEquations(N, vars, scalar_params, reactions, globals), N_span, vars_ini, options);
    
    % save Y_eta and vars_N
    Y_eta(i,:) = horzcat(vars_sol(end,4), vars_sol(end,8:end), which_model, scalar_params); % [eta, Y_i, which_model, scalar_params]
    
    if i == i_Planck
        model_info           = zeros(length(N_sol), 1+length(scalar_params));
        model_info(1, 1)     = which_model;
        model_info(1, 2:end) = scalar_params;
        vars_N = horzcat(N_sol, vars_sol, model_info);
        dlmwrite('vars_N.dat', vars_N, 'delimiter', '\t', 'precision', '%.16E');
    end
    
    send(data_queue, i);
end
dlmwrite('Y_eta.dat', Y_eta, 'delimiter', '\t', 'precision', '%.16E');
close(wait_bar);
delete(gcp('nocreate'));

PlotFigures(globals);
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Three main functions in BBNLab.m: InitialConditions, EvolutionEquations and PlotFigures
%%%% Other independent functions:      InputPhysicalData, ScalarFields and SumReactionRates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [vars_ini, V_null_update] = InitialConditions(T_photon, eta, scalar_params, globals)
    % we omit the subscript ini
    
    CL                 = globals.CL;
    GN                 = globals.GN;
    KB                 = globals.KB;
    ME                 = globals.ME;
    N_nuclei           = globals.N_nuclei;
    proton_number_list = globals.proton_number_list;
    i_bessel           = globals.i_bessel;
    
    z_photon   = ME*CL^2./(KB*T_photon);
    T_neutrino = T_photon;
    z_neutrino = z_photon;
    [rho_photon, ~, n_photon] = PhotonGas(T_photon, globals);
    n_baryon = n_photon * eta;
    
    Y_list = zeros(N_nuclei, 1);
    [Y_list(1), Y_list(2)] = NeutronProtonEquilibrium(T_photon, globals);
    
    sum_i_L = sum((-1).^(i_bessel+1) .* i_bessel .* FunctionL(i_bessel.*z_photon));
    phi_electron = zeta(3) * eta * sum(proton_number_list.*Y_list) / (z_photon^3*sum_i_L);
    
    rho_neutrino      = NeutrinoGas(T_neutrino, globals);
    [rho_electron, ~] = ElectronGas(T_photon, phi_electron, globals);
    [rho_baryon, ~]   = BaryonGas(T_photon, n_baryon, Y_list, globals);
    rho_matter        = rho_neutrino + rho_photon + rho_electron + rho_baryon;
    
    scalar_field.params         = scalar_params;
    scalar_field.phi            = 0; % may can be set as NaN
    scalar_field.phi_dot        = 0;
    scalar_field.whether_ini    = 1; % see ScalarFields.m
    scalar_field.rho_matter_ini = rho_matter;
    [~, ini_output] = ScalarFields(scalar_field, globals);
    [phi_scalar, phi_scalar_dot, V_null_update] = deal(ini_output(1), ini_output(2), ini_output(3));
    
    scalar_field.params         = scalar_params;
    scalar_field.phi            = phi_scalar;
    scalar_field.phi_dot        = phi_scalar_dot;
    scalar_field.whether_ini    = 0;
    scalar_field.rho_matter_ini = NaN;
    [evol_output, ~]            = ScalarFields(scalar_field, globals);
    rho_phi                     = evol_output(1);
    rho_total   = rho_phi + rho_matter;
    Hubble_rate = sqrt(8*pi*GN/3*rho_total);
    t_cosmic    = 1/(2*Hubble_rate);
    
    vars_ini        = zeros(N_nuclei+7, 1); % use column vectors whenever possible
    vars_ini(1)     = z_photon;
    vars_ini(2)     = z_neutrino;
    vars_ini(3)     = phi_electron;
    vars_ini(4)     = eta;
    vars_ini(5)     = t_cosmic;
    vars_ini(6)     = phi_scalar;
    vars_ini(7)     = phi_scalar_dot;
    vars_ini(8:end) = Y_list;
end

function dvarsdN = EvolutionEquations(N, vars, scalar_params, reactions, globals)

    CL                 = globals.CL;
    GN                 = globals.GN;
    HB                 = globals.HB;
    KB                 = globals.KB;
    ME                 = globals.ME;
    N_nuclei           = globals.N_nuclei;
    mass_list          = globals.mass_list;
    proton_number_list = globals.proton_number_list;
    i_bessel           = globals.i_bessel;
    
    z_photon       = vars(1);
    z_neutrino     = vars(2);
    phi_electron   = vars(3);
    eta            = vars(4);
    t_cosmic       = vars(5); % for intuition if want
    phi_scalar     = vars(6);
    phi_scalar_dot = vars(7);
    Y_list         = vars(8:end);
    
    T_photon   = ME*CL^2/(KB*z_photon);
    T_neutrino = ME*CL^2/(KB*z_neutrino);
    [rho_photon, p_photon, n_photon] = PhotonGas(T_photon, globals);
    [rho_electron, p_electron]       = ElectronGas(T_photon, phi_electron, globals);
    n_baryon                         = eta * n_photon;
    [rho_baryon, p_baryon]           = BaryonGas(T_photon, n_baryon, Y_list, globals);
    Gamma_list = SumReactionRates(T_photon, T_neutrino, n_baryon, Y_list, reactions, globals);
    
    scalar_field.params         = scalar_params;
    scalar_field.phi            = phi_scalar;
    scalar_field.phi_dot        = phi_scalar_dot;
    scalar_field.whether_ini    = 0;
    scalar_field.rho_matter_ini = NaN;
    [evol_output, ~]            = ScalarFields(scalar_field, globals);
    [rho_phi, V_prime]          = deal(evol_output(1), evol_output(2));
    
    rho_neutrino = NeutrinoGas(T_neutrino, globals);
    rho_total    = rho_phi + rho_neutrino + rho_photon + rho_electron + rho_baryon;
    Hubble_rate  = sqrt(8*pi*GN/3*rho_total);
    
    cosh_bessel = cosh(i_bessel.*phi_electron);
    sinh_bessel = sinh(i_bessel.*phi_electron);
    
    kappa_1 = SumNaN((-1).^(i_bessel+1) .* i_bessel .* cosh_bessel .* FunctionL(i_bessel.*z_photon));
    kappa_2 = SumNaN((-1).^(i_bessel+1) .* i_bessel .* sinh_bessel .* FunctionLPrime(i_bessel.*z_photon));
    kappa_3 = pi^2*HB^3*n_baryon/(2*ME^3*CL^3)*sum(proton_number_list.*(Gamma_list-3*Hubble_rate*Y_list));
    kappa_4 = SumNaN((-1).^(i_bessel+1) .* i_bessel .* sinh_bessel .* FunctionM(i_bessel.*z_photon));
    kappa_5 = SumNaN((-1).^(i_bessel+1) .* i_bessel .* cosh_bessel .* FunctionMPrime(i_bessel.*z_photon)) ...
              + pi^2*HB^3/(2*ME^4*CL^3)*(- 4*rho_photon/z_photon - 3*ME*n_baryon/(2*z_photon^2)*sum(Y_list));
    p_plasma = p_photon + p_electron + p_baryon;
    kappa_6 = pi^2*HB^3/(2*ME^4*CL^3) * (- 3*Hubble_rate*(rho_photon+rho_electron+p_plasma/CL^2) - n_baryon*sum(Gamma_list.*(mass_list+3*ME/(2*z_photon)))); 
    
    dvarsdt        = zeros(N_nuclei+7, 1);
    dvarsdt(1)     = (kappa_1*kappa_6-kappa_3*kappa_4) / (kappa_1*kappa_5-kappa_2*kappa_4);
    dvarsdt(2)     = Hubble_rate * z_neutrino;
    dvarsdt(3)     = (kappa_3*kappa_5-kappa_2*kappa_6) / (kappa_1*kappa_5-kappa_2*kappa_4);
    dvarsdt(4)     = 3*pi^2*HB^3*z_photon^2*n_baryon/(2*zeta(3)*ME^3*CL^3) * (dvarsdt(1) - Hubble_rate*z_photon);
    dvarsdt(5)     = 1; % dt/dt=1
    dvarsdt(6)     = phi_scalar_dot;
    dvarsdt(7)     = - 3*Hubble_rate*phi_scalar_dot - CL^2*V_prime;
    dvarsdt(8:end) = Gamma_list;
    
    dvarsdN = dvarsdt./Hubble_rate;
end

function [] = PlotFigures(globals)
    fprintf('Ploting the figures, please wait ... ')
    
    CL                 = globals.CL;
    KB                 = globals.KB;
    ME                 = globals.ME;
    i_bessel           = globals.i_bessel;
    mass_number_list   = globals.mass_number_list;
    proton_number_list = globals.proton_number_list;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% one BBN evolution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the 1st    column  is  N
    % the 2-8th  columns are cosmic background dynamics (including thermodynamics)
    % the 9-17th columns are Y_list
    % the 18th   column  are which_model
    % the next           is  scalar_params
    vars_N         = load('vars_N.dat');
    N_folding      = vars_N(:,1);
    z_photon       = vars_N(:,2);
    z_neutrino     = vars_N(:,3);
    phi_electron   = vars_N(:,4);
    eta            = vars_N(:,5);
    t_cosmic       = vars_N(:,6);
    phi_scalar     = vars_N(:,7);
    phi_scalar_dot = vars_N(:,8);
    Y_neutron      = abs(vars_N(:,9));  % abs is for the following loglog plot (In principle Y_i>=0. But numerical errors exist!)
    Y_proton       = abs(vars_N(:,10));
    Y_D            = abs(vars_N(:,11));
    Y_T            = abs(vars_N(:,12));
    Y_He3          = abs(vars_N(:,13));
    Y_He4          = abs(vars_N(:,14));
    Y_Li6          = abs(vars_N(:,15));
    Y_Li7          = abs(vars_N(:,16));
    Y_Be7          = abs(vars_N(:,17));
    scalar_params  = vars_N(1,19:end);
    
    T_photon       = ME * CL^2 ./ (KB * z_photon);
    T_neutrino     = ME * CL^2 ./ (KB * z_neutrino);
    N_sample       = length(T_photon);
    
    Y_neutron_equi = NaN(N_sample,1);
    Y_proton_equi  = NaN(N_sample,1);
    Y_D_equi       = NaN(N_sample,1);
    Y_T_equi       = NaN(N_sample,1);
    Y_He3_equi     = NaN(N_sample,1);
    Y_He4_equi     = NaN(N_sample,1);
    Y_Li6_equi     = NaN(N_sample,1);
    Y_Li7_equi     = NaN(N_sample,1);
    Y_Be7_equi     = NaN(N_sample,1);
    for i = 1 : N_sample
        [Y_neutron_equi(i), Y_proton_equi(i)] = NeutronProtonEquilibrium(T_photon(i), globals);
        if T_photon(i) > 5E+9 && T_photon(i) < 1E10 
            Y_heavy     = HeavyNucleiEquilibrium(T_photon(i), eta(i), Y_neutron(i), Y_proton(i), globals);
            Y_D_equi(i)   = Y_heavy(1);
            Y_T_equi(i)   = Y_heavy(2);
            Y_He3_equi(i) = Y_heavy(3);
            Y_He4_equi(i) = Y_heavy(4);
            Y_Li6_equi(i) = Y_heavy(5);
            Y_Li7_equi(i) = Y_heavy(6);
            Y_Be7_equi(i) = Y_heavy(7);
        end
    end
    
    rho_neutrino = zeros(N_sample,1);
    rho_photon   = zeros(N_sample,1);
    p_photon     = zeros(N_sample,1);
    n_photon     = zeros(N_sample,1);
    rho_electron = zeros(N_sample,1);
    p_electron   = zeros(N_sample,1);
    rho_baryon   = zeros(N_sample,1);
    p_baryon     = zeros(N_sample,1);
    n_baryon     = zeros(N_sample,1);
    rho_phi      = zeros(N_sample,1);
    for i = 1 : N_sample
        rho_neutrino(i) = NeutrinoGas(T_neutrino(i), globals);
        [rho_photon(i), p_photon(i), n_photon(i)] = PhotonGas(T_photon(i), globals);
        [rho_electron(i), p_electron(i)] = ElectronGas(T_photon(i), phi_electron(i), globals);
        n_baryon(i) = n_photon(i)*eta(i);
        Y_list = abs(vars_N(i, 9:17))';
        [rho_baryon(i), p_baryon(i)] = BaryonGas(T_photon(i), n_baryon(i), Y_list, globals);
        scalar_field.params         = scalar_params;
        scalar_field.phi            = phi_scalar(i);
        scalar_field.phi_dot        = phi_scalar_dot(i);
        scalar_field.whether_ini    = 0;
        scalar_field.rho_matter_ini = NaN;
        [evol_output, ~]            = ScalarFields(scalar_field, globals);
        rho_phi(i)                  = evol_output(1);
    end
    rho_total   = rho_phi + rho_neutrino + rho_photon + rho_electron + rho_baryon;
    
    phi_electron_hot  = NaN(N_sample, 1);
    phi_electron_cold = NaN(N_sample, 1);
    for i = 1 : N_sample
        Y_list = abs(vars_N(i, 9:17))';
        if T_photon(i) > 0 
            sum_i_L = sum((-1).^(i_bessel+1) .* i_bessel .* FunctionL(i_bessel.*z_photon(i)));
            phi_electron_hot(i) = zeta(3) * eta(i) * sum(proton_number_list.*Y_list) / (z_photon(i)^3*sum_i_L);
        end
        if T_photon(i) < 3E+8
            phi_electron_cold(i) = z_photon(i) + log(sqrt(2/pi) * 2*zeta(3)*eta(i)/z_photon(i)^(3/2) * sum(proton_number_list.*Y_list));
            phi_electron_cold(phi_electron_cold<0) = NaN;
        end
    end
    
    p_neutrino = 1/3*rho_neutrino*CL^2;
    p_matter = p_neutrino + p_photon + p_electron + p_baryon;
    rho_matter = rho_neutrino + rho_photon + rho_electron + rho_baryon;
    w_matter = p_matter./(rho_matter*CL^2);
    
    % verify electrical neutrality
    charge_negative      = NaN(N_sample,1);
    charge_negative_cold = NaN(N_sample,1);
    charge_positive      = NaN(N_sample,1);
    for i = 1 : N_sample
        charge_negative(i) = z_photon(i)^3/zeta(3) * SumNaN((-1).^(i_bessel+1) .* sinh(i_bessel.*phi_electron(i)) .* FunctionL(i_bessel.*z_photon(i)));
        if T_photon(i) < 1E+8
            charge_negative_cold(i) = z_photon(i)^3/zeta(3) * SumNaN((-1).^(i_bessel+1) .* sinh(i_bessel.*phi_electron_cold(i)) .* FunctionL(i_bessel.*z_photon(i)));
        end
        Y_list = abs(vars_N(i, 9:17))';
        charge_positive(i) = eta(i) * sum(proton_number_list.*Y_list);
    end
    
    % verify energy conservation
    rho_plasma         = rho_photon + rho_electron + rho_baryon;
    p_plasma           = p_photon + p_electron + p_baryon;
    delta_N            = 1E-4;
    N_folding_interp   = transpose(min(N_folding) : delta_N : max(N_folding));
    rho_plasma_interp  = interp1(N_folding, rho_plasma, N_folding_interp, 'spline');
    p_plasma_interp    = interp1(N_folding, p_plasma, N_folding_interp, 'spline');
    rho_plasma_dN      = gradient(rho_plasma_interp, delta_N);
    rho_plasma_Hubble  = 3*(rho_plasma_interp+p_plasma_interp/CL^2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Y-eta %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y_eta = load('Y_eta.dat');
    [No_n, No_p, No_D, No_T, No_He3, No_He4, No_Li6, No_Li7, No_Be7] = deal(1,2,3,4,5,6,7,8,9);
    
    eta_list     = Y_eta(:,1);
    YE_proton    = Y_eta(:,No_p+1);
    YE_deuterium = Y_eta(:,No_D+1);
    YE_He3       = Y_eta(:,No_He3+1) + Y_eta(:,No_T+1);
    YE_He4       = Y_eta(:,No_He4+1);
    YE_Li7       = Y_eta(:,No_Li7+1) + Y_eta(:,No_Be7+1);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% plot the result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(1)
    subplot(331);
    loglog(T_photon, Y_D, '-', T_photon, Y_D_equi, '--')
    hold on
    loglog(T_photon, Y_neutron)
    loglog(T_photon, Y_proton)
    loglog(T_photon, Y_T, '-', T_photon, Y_T_equi, '--')
    loglog(T_photon, Y_He3, '-', T_photon, Y_He3_equi, '--')
    loglog(T_photon, Y_He4, '-', T_photon, Y_He4_equi, '--')
    loglog(T_photon, Y_Li6, '-', T_photon, Y_Li6_equi, '--')
    loglog(T_photon, Y_Li7, '-', T_photon, Y_Li7_equi, '--')
    loglog(T_photon, Y_Be7, '-', T_photon, Y_Be7_equi, '--')
    hold off
    axis([min(T_photon), max(T_photon), 1E-65, 1])
    legend('Actual', 'Equilibrium', 'Location', 'SouthEast')
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('$Y_i$', 'Interpreter', 'LaTex')
    % title('Evolution of $Y_i(T)$ and the equilibrium values of heavy nuclei', 'Interpreter', 'LaTex')
    
    % repeat Smith1993 Fig. 1 https://doi.org/10.1086/191763
    subplot(332)
    loglog(T_photon, (Y_neutron+Y_D+2*Y_T+Y_He3+2*Y_He4+3*Y_Li6+4*Y_Li7+3*Y_Be7)./(Y_proton+Y_D+Y_T+2*Y_He3+2*Y_He4+3*Y_Li6+3*Y_Li7+4*Y_Be7))
    hold on
    loglog(T_photon, Y_neutron_equi./Y_proton_equi)
    hold off
    axis([min(T_photon), max(T_photon), 0.1, 1])
    legend('Actual (all nuclei)', 'Equilibrium (only neutron and proton)', 'Location', 'NorthWest')
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('Neutron-Proton Ratio', 'Interpreter', 'LaTex')
    
    subplot(333)
    loglog(T_photon, phi_electron, '-')
    hold on
    loglog(T_photon, phi_electron_hot, '--')
    loglog(T_photon, phi_electron_cold, '--')
    hold off
    axis([min(T_photon), max(T_photon), min(phi_electron)/2, max(phi_electron)*2])
    legend('Actual', 'High temperature approximation', 'Low temperature approximation', 'Location', 'NorthEast')
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('$\phi_e$', 'Interpreter', 'LaTex')
    
    subplot(334)
    loglog(T_photon, rho_phi./rho_total)
    axis([min(T_photon), max(T_photon), min(rho_phi./rho_total)/1.25, 1.1])
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('$\rho_\phi/\rho_{\rm total}$', 'Interpreter', 'LaTex')
    
    subplot(335)
    loglog(T_photon, w_matter);
    axis([min(T_photon), max(T_photon), 0.29, 0.34])
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('EOS of matter', 'Interpreter', 'LaTex')
    
    subplot(336)
    semilogy(N_folding, T_photon, '-', N_folding, T_neutrino, '--')
    xlim([min(N_folding), max(N_folding)])
    legend('$T_\gamma$', '$T_\nu$', 'Interpreter', 'LaTex', 'Location', 'NorthEast')
    xlabel('The e-folding number $N$', 'Interpreter', 'LaTex')
    ylabel('Temperature [Kelvin]', 'Interpreter', 'LaTex')
    
    subplot(337)
    sum_AiYi = mass_number_list(1)*Y_neutron + mass_number_list(2)*Y_proton + mass_number_list(3)*Y_D + mass_number_list(4)*Y_T + mass_number_list(5)*Y_He3 ...
        + mass_number_list(6)*Y_He4 + mass_number_list(7)*Y_Li6 + mass_number_list(8)*Y_Li7 + mass_number_list(9)*Y_Be7; % should be equal to 1
    loglog(T_photon, abs(sum_AiYi-1))
    axis([min(T_photon), max(T_photon), 1E-17, 1E-7])
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('$\left|1-\sum_iA_iY_i\right|$', 'Interpreter', 'LaTex')
    
    subplot(338)
    loglog(T_photon, charge_negative, '-');
    hold on
    loglog(T_photon, charge_negative_cold, '-'); % this plot shows charge_negative's low temperature error may due to phi_e's "minor error"
    loglog(T_photon, charge_positive, '--');
    hold off
    axis([min(T_photon), max(T_photon), 4E-10, 1E-7])
    legend('negative (left)', 'negative ($\phi_{e,{\rm cold}}$)', 'positive (right)', 'Interpreter', 'LaTex')
    xlabel('Photon temperature $T\ {\rm [Kelvin]}$', 'Interpreter', 'LaTex')
    ylabel('$+/-$ charges', 'Interpreter', 'LaTex')
    
    subplot(339)
    semilogy(N_folding_interp, -rho_plasma_dN, '-')
    hold on
    semilogy(N_folding_interp, rho_plasma_Hubble, '--')
    semilogy(N_folding_interp, abs((rho_plasma_dN+rho_plasma_Hubble)./rho_plasma_Hubble))
    hold off
    xlim([min(N_folding_interp), max(N_folding_interp)])
    legend('$-{\rm d}\rho_{\gamma eb}/{\rm d}N$', '$3(\rho_{\gamma eb}+p_{\gamma eb}/c^2)$', ...
        '$\left|1+\frac{{\rm d}\rho_{\gamma eb}/{\rm d}N}{3(\rho_{\gamma eb}+p_{\gamma eb}/c^2)}\right|$', 'Interpreter', 'LaTex')
    xlabel('The e-folding number $N$', 'Interpreter', 'LaTex')
    ylabel('density gradient', 'Interpreter', 'LaTex')
    
    figure(2)
    loglog(eta_list, 4*YE_He4)
    hold on
    loglog(eta_list, YE_deuterium./YE_proton)
    loglog(eta_list, YE_He3./YE_proton)
    loglog(eta_list, YE_Li7./YE_proton)
    fill([min(eta_list) max(eta_list) max(eta_list) min(eta_list)], [0.2419 0.2419 0.2487 0.2487],       'g', 'FaceAlpha', 0.5, 'LineStyle','none') % observations
    fill([min(eta_list) max(eta_list) max(eta_list) min(eta_list)], [2.4970 2.4970 2.5570 2.5570]*1E-5,  'g', 'FaceAlpha', 0.5, 'LineStyle','none')
    fill([min(eta_list) max(eta_list) max(eta_list) min(eta_list)], [0.9    0.9    1.3    1.3   ]*1E-5,  'g', 'FaceAlpha', 0.5, 'LineStyle','none')
    fill([min(eta_list) max(eta_list) max(eta_list) min(eta_list)], [1.30   1.30   1.93   1.93  ]*1E-10, 'g', 'FaceAlpha', 0.5, 'LineStyle','none')
    fill([6.0991E-10 6.1758E-10 6.1758E-10 6.0991E-10], [1E-20 1E-20 1E10 1E10], 'k', 'FaceAlpha', 0.2, 'LineStyle','none')
    plots = get(gca,'Children');
    set(gca, 'Children', [plots(9), plots(8), plots(7), plots(6) plots(5) plots(4) plots(3) plots(2) plots(1)])
    hold off
    legend(plots([9 8 7 6 5 1]), '4Y_{He4}', 'D/H', 'He3/H', 'Li7/H', 'Observations', 'Planck result', 'Location', 'NorthWest')
    axis([min(eta_list)/1.0001, max(eta_list), 1E-11, 1])
    xlabel('Baryon-to-photon ratio $\eta_{\rm today}$', 'Interpreter', 'LaTex')
    
    fprintf('Finished.\n')
end

function [Y_neutron, Y_proton] = NeutronProtonEquilibrium(T_photon, globals)

    CL = globals.CL;
    KB = globals.KB;
    MN = globals.MN;
    MP = globals.MP;
    
    Delta = MN - MP;
    beta = 1/(KB*T_photon);
    
    Y_neutron = 1/(1 + (1-3*Delta/(2*MN))*exp(beta*Delta*CL^2));
    Y_proton  = 1 - Y_neutron;
end

function Y_heavy = HeavyNucleiEquilibrium(T_photon, eta, Y_neutron, Y_proton, globals)

    CL                 = globals.CL;
    KB                 = globals.KB;
    MN                 = globals.MN;
    MP                 = globals.MP;
    mass_list          = globals.mass_list;
    mass_number_list   = globals.mass_number_list;
    proton_number_list = globals.proton_number_list;
    spin_list          = globals.spin_list;
    
    beta = 1/(KB*T_photon);
    
    spin_degeneracy_list = 2*spin_list + 1;
    neutron_number_list  = mass_number_list - proton_number_list;
    binding_energy_list  = proton_number_list*MP + neutron_number_list*MN - mass_list;
    
    Y_list = spin_degeneracy_list .* 2.^((3*mass_number_list-5)/2) .* pi.^((1-mass_number_list)/2) .* zeta(3).^(mass_number_list-1);
    Y_list = Y_list .* CL.^(3*(1-mass_number_list)) .* eta.^(mass_number_list-1);
    Y_list = Y_list .* Y_proton.^proton_number_list .* Y_neutron.^neutron_number_list;
    Y_list = Y_list .* (mass_list.*beta.^(1-mass_number_list)./(MP.^proton_number_list.*MN.^neutron_number_list)).^(3/2);
    Y_list = Y_list .* exp(beta.*binding_energy_list.*CL^2);
    
    Y_heavy = Y_list(3:end);
end

function rho_neutrino = NeutrinoGas(T_neutrino, globals)
    
    CL = globals.CL;
    HB = globals.HB;
    KB = globals.KB;
    
    N_neutrino_eff = 3.046;
    rho_neutrino = 7*N_neutrino_eff*pi^2*KB^4*T_neutrino^4/(120*HB^3*CL^5);
end

function [rho_photon, p_photon, n_photon] = PhotonGas(T_photon, globals)

    CL = globals.CL;
    HB = globals.HB;
    KB = globals.KB;
    
    rho_photon = pi^2*KB^4*T_photon^4/(15*HB^3*CL^5);
    p_photon   = 1/3*rho_photon*CL^2;
    n_photon   = 2*zeta(3)*KB^3*T_photon^3/(pi^2*HB^3*CL^3);
end

function [rho_electron, p_electron] = ElectronGas(T_photon, phi_electron, globals)

    CL       = globals.CL;
    HB       = globals.HB;
    KB       = globals.KB;
    ME       = globals.ME;
    i_bessel = globals.i_bessel;
    
    z_photon     = ME*CL^2/(KB*T_photon);
    cosh_bessel  = cosh(i_bessel.*phi_electron);
    rho_electron = 2*ME^4*CL^3/(pi^2*HB^3) * SumNaN((-1).^(i_bessel+1) .* cosh_bessel .* FunctionM(i_bessel.*z_photon));
    p_electron   = 2*ME^3*CL^3*KB*T_photon/(pi^2*HB^3) * SumNaN((-1).^(i_bessel+1) ./ i_bessel .* cosh_bessel .* FunctionL(i_bessel.*z_photon));
end

function [rho_baryon, p_baryon] = BaryonGas(T_photon, n_baryon, Y_list, globals)

    CL        = globals.CL;
    KB        = globals.KB;
    ME        = globals.ME;
    mass_list = globals.mass_list;
    
    z_photon   = ME*CL^2/(KB*T_photon);
    rho_baryon = n_baryon * sum(Y_list.*(mass_list+3*ME/(2*z_photon)));
    p_baryon   = n_baryon*KB*T_photon*sum(Y_list);
end

function L = FunctionL(x)
    L = besselk(2,x)./x; % see 'help besselk' in Matlab
end

function M = FunctionM(x)
    M = 1./x.*(3/4*besselk(3,x)+1/4*besselk(1,x));
end

function L_prime = FunctionLPrime(x)
    L_prime = - (1./x+6./x.^3).*besselk(1,x) - 3./x.^2.*besselk(0,x);
end

function M_prime = FunctionMPrime(x)
    M_prime = - (24./x.^4+5./x.^2).*besselk(1,x) - (12./x.^3+1./x).*besselk(0,x);
end

function x_sum = SumNaN(x)
    % In Matlab, a value larger than realmax (1.7977E+308) will be identified as Inf.
    % However Inf*0 = NaN may cause an error. 
    % We use SumNaN to deal with this issue.
    x(isnan(x)) = 0;
    x_sum = sum(x);
end

function UpdateWaitBar(wait_bar, start_time)
    parfor_progress    = wait_bar.UserData;
    parfor_progress(1) = parfor_progress(1) + 1;
    progress_percent   = parfor_progress(1)/parfor_progress(2);
    waitbar(progress_percent, wait_bar, sprintf('start at %s, and %d%% done', start_time, floor(progress_percent*100)));
    wait_bar.UserData  = parfor_progress;
end
