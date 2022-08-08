function [globals, reactions] = InputPhysicalData(which_model)

    globals.which_model = which_model;
    
    globals.i_bessel = linspace(1, 20, 20);
    
    %--------------------- SI units ----------------------
    globals.Kelvin   = 1; % temperature unit
    globals.kilogram = 1; % mass unit
    globals.meter    = 1; % length unit
    globals.second   = 1; % time unit
    globals.Joule    = globals.kilogram * globals.meter^2 * globals.second^(-2); % energy unit
    globals.MeV      = 1.60217663E-13   * globals.Joule; % energy unit in MeV
    
    %--------------------- Values of physical constants -----------------------
    globals.CL = 2.99792458E+08 * globals.meter   / globals.second; % speed of light
    globals.GN = 6.67430000E-11 * globals.meter^3 * globals.kilogram^(-1) * globals.second^(-2); % Newtonian gravitational constant
    globals.HB = 1.05457182E-34 * globals.Joule   * globals.second; % reduced Planck constant
    globals.KB = 1.38064852E-23 * globals.Joule   / globals.Kelvin; % Boltzmann constant
    globals.ME = 9.10938356E-31 * globals.kilogram; % Electron mass in kilogram
    globals.MN = 1.67492750E-27 * globals.kilogram; % Neutron mass in kilogram
    globals.MP = 1.67262192E-27 * globals.kilogram; % Proton mass in kilogram
    globals.MU = 1.66053907E-27 * globals.kilogram; % atomic mass constant
    
    %--------------------- the nuclear reaction network -------------------
    globals.N_nuclei    = 9;
    globals.N_reactions = 32;
    
    [No_n, No_p, No_D, No_T, No_He3, No_He4, No_Li6, No_Li7, No_Be7] = deal(1,2,3,4,5,6,7,8,9);
    globals.mass_excess_list   = [8.0713 7.2890 13.1357 14.9498 14.9312 2.4249 14.0869 14.9071 15.7690]' * globals.MeV / globals.CL^2;
    globals.mass_number_list   = [1      1      2       3       3       4      6       7       7]';
    globals.proton_number_list = [0      1      1       1       2       2      3       3       4]';
    globals.spin_list          = [1/2    1/2    1       1/2     1/2     0      1       3/2     3/2]';
    globals.mass_list          = globals.mass_number_list.*globals.MU + globals.mass_excess_list - globals.proton_number_list.*globals.ME;
    
    %-------------------- reaction_matrix -------------------
    reaction_matrix = zeros(globals.N_reactions, globals.N_nuclei);
    
    %------------- (1st) n -> p
    reaction_matrix(1, No_n) = -1;
    reaction_matrix(1, No_p) =  1;
    
    %------------- (2nd) p + n -> g + D 
    reaction_matrix(2, No_p) = -1;
    reaction_matrix(2, No_n) = -1;
    reaction_matrix(2, No_D) =  1;

    %------------- (3rd) D + n -> g + T 
    reaction_matrix(3, No_D) = -1;
    reaction_matrix(3, No_n) = -1;
    reaction_matrix(3, No_T) =  1;
    
    %------------- (4th) D + p -> g + He3 
    reaction_matrix(4, No_D)   = -1;
    reaction_matrix(4, No_p)   = -1;
    reaction_matrix(4, No_He3) =  1;
    
    %------------- (5th) D + D -> n + He3 
    reaction_matrix(5, No_D)   = -2;
    reaction_matrix(5, No_n)   =  1;
    reaction_matrix(5, No_He3) =  1;
    
    %------------- (6th) D + D -> p + T
    reaction_matrix(6, No_D) = -2;
    reaction_matrix(6, No_p) =  1;
    reaction_matrix(6, No_T) =  1;
    
    %------------- (7th) T -> v + e + He3
    reaction_matrix(7, No_T)   = -1;
    reaction_matrix(7, No_He3) =  1;
    
    %------------- (8th) T + p -> g + He4
    reaction_matrix(8, No_T)   = -1;
    reaction_matrix(8, No_p)   = -1;
    reaction_matrix(8, No_He4) =  1;
    
    %------------- (9th) T + D -> n + He4
    reaction_matrix(9, No_T)   = -1;
    reaction_matrix(9, No_D)   = -1;
    reaction_matrix(9, No_n)   =  1;
    reaction_matrix(9, No_He4) =  1;
    
    %------------- (10th) T + T -> n + n + He4
    reaction_matrix(10, No_T)   = -2;
    reaction_matrix(10, No_n)   =  2;
    reaction_matrix(10, No_He4) =  1;
    
    %------------- (11th) He3 + n -> g + He4
    reaction_matrix(11, No_He3) = -1;
    reaction_matrix(11, No_n)   = -1;
    reaction_matrix(11, No_He4) =  1;
    
    %------------- (12th) He3 + n -> p + T
    reaction_matrix(12, No_He3) = -1;
    reaction_matrix(12, No_n)   = -1;
    reaction_matrix(12, No_p)   =  1;
    reaction_matrix(12, No_T)   =  1;
    
    %------------- (13th) He3 + D -> p + He4
    reaction_matrix(13, No_He3) = -1;
    reaction_matrix(13, No_D)   = -1;
    reaction_matrix(13, No_p)   =  1;
    reaction_matrix(13, No_He4) =  1;
    
    %------------- (14th) He3 + T -> g + Li6
    reaction_matrix(14, No_He3) = -1;
    reaction_matrix(14, No_T)   = -1;
    reaction_matrix(14, No_Li6) =  1;
    
    %------------- (15th) He3 + T -> D + He4
    reaction_matrix(15, No_He3) = -1;
    reaction_matrix(15, No_T)   = -1;
    reaction_matrix(15, No_D)   =  1;
    reaction_matrix(15, No_He4) =  1;
    
    %------------- (16th) He3 + T -> n + p + He4
    reaction_matrix(16, No_He3) = -1;
    reaction_matrix(16, No_T)   = -1;
    reaction_matrix(16, No_n)   =  1;
    reaction_matrix(16, No_p)   =  1;
    reaction_matrix(16, No_He4) =  1;
    
    %------------- (17th) He3 + He3 -> p + p + He4
    reaction_matrix(17, No_He3) = -2;
    reaction_matrix(17, No_p)   =  2;
    reaction_matrix(17, No_He4) =  1;
    
    %------------- (18th) He4 + D -> g + Li6
    reaction_matrix(18, No_He4) = -1;
    reaction_matrix(18, No_D)   = -1;
    reaction_matrix(18, No_Li6) =  1;
    
    %------------- (19th) He4 + T -> g + Li7
    reaction_matrix(19, No_He4) = -1;
    reaction_matrix(19, No_T)   = -1;
    reaction_matrix(19, No_Li7) =  1;
    
    %------------- (20th) He4 + He3 -> g + Be7
    reaction_matrix(20, No_He4) = -1;
    reaction_matrix(20, No_He3) = -1;
    reaction_matrix(20, No_Be7) =  1;
    
    %------------- (21th) Li6 + n -> g + Li7
    reaction_matrix(21, No_Li6) = -1;
    reaction_matrix(21, No_n)   = -1;
    reaction_matrix(21, No_Li7) =  1;
    
    %------------- (22th) Li6 + n -> T + He4
    reaction_matrix(22, No_Li6) = -1;
    reaction_matrix(22, No_n)   = -1;
    reaction_matrix(22, No_T)   =  1;
    reaction_matrix(22, No_He4) =  1;
    
    %------------- (23th) Li6 + p -> g + Be7
    reaction_matrix(23, No_Li6) = -1;
    reaction_matrix(23, No_p)   = -1;
    reaction_matrix(23, No_Be7) =  1;
    
    %------------- (24th) Li6 + p -> He3 + He4
    reaction_matrix(24, No_Li6) = -1;
    reaction_matrix(24, No_p)   = -1;
    reaction_matrix(24, No_He3) =  1;
    reaction_matrix(24, No_He4) =  1;
    
    %------------- (25th) Li6 + D -> n + Be7
    reaction_matrix(25, No_Li6) = -1;
    reaction_matrix(25, No_D)   = -1;
    reaction_matrix(25, No_n)   =  1;
    reaction_matrix(25, No_Be7) =  1;
    
    %------------- (26th) Li6 + D -> p + Li7
    reaction_matrix(26, No_Li6) = -1;
    reaction_matrix(26, No_D)   = -1;
    reaction_matrix(26, No_p)   =  1;
    reaction_matrix(26, No_Li7) =  1;
    
    %------------- (27th) Li7 + p -> He4 + He4
    reaction_matrix(27, No_Li7) = -1;
    reaction_matrix(27, No_p)   = -1;
    reaction_matrix(27, No_He4) =  2;
    
    %------------- (28th) Li7 + p -> g + He4 + He4
    reaction_matrix(28, No_Li7) = -1;
    reaction_matrix(28, No_p)   = -1;
    reaction_matrix(28, No_He4) =  2;
    
    %------------- (29th) Li7 + D -> n + He4 + He4
    reaction_matrix(29, No_Li7) = -1;
    reaction_matrix(29, No_D)   = -1;
    reaction_matrix(29, No_n)   =  1;
    reaction_matrix(29, No_He4) =  2;
    
    %------------- (30th) Be7 + n -> p + Li7
    reaction_matrix(30, No_Be7) = -1;
    reaction_matrix(30, No_n)   = -1;
    reaction_matrix(30, No_p)   =  1;
    reaction_matrix(30, No_Li7) =  1;
    
    %------------- (31th) Be7 + n -> He4 + He4
    reaction_matrix(31, No_Be7) = -1;
    reaction_matrix(31, No_n)   = -1;
    reaction_matrix(31, No_He4) =  2;
    
    %------------- (32th) Be7 + D -> p + He4 + He4
    reaction_matrix(32, No_Be7) = -1;
    reaction_matrix(32, No_D)   = -1;
    reaction_matrix(32, No_p)   =  1;
    reaction_matrix(32, No_He4) =  2;
    
    reactions.reaction_matrix = reaction_matrix;
    
    reactions.log_forward_rates      = log(load('.\ReactionRates\NuclearForwardRates.dat'));
    reactions.nuclear_inverse_params = load('.\ReactionRates\NuclearInverseParams.dat');
end
