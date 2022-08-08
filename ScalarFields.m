function [evol_output, ini_output] = ScalarFields(scalar_field, globals)
    % output : evol_output = [rho_phi V_prime] for the evolution equations
    % output : ini_output  = [phi_ini phi_dot_ini V_null_update] for the initial conditions
    
    CL          = globals.CL;
    GN          = globals.GN;
    which_model = globals.which_model;
    
    % scalar_field is a structure, includes 5 variables
    scalar_params  = scalar_field.params;
    phi            = scalar_field.phi;
    phi_dot        = scalar_field.phi_dot;
    whether_ini    = scalar_field.whether_ini;
    rho_matter_ini = scalar_field.rho_matter_ini;
    
    % no scalar field
    if which_model == 1
        evol_output  = [0 0];
        ini_output   = [0 0 0];
    end
    
    % exponential scalar field
    if which_model == 2
        [lambda, V_null] = deal(scalar_params(1), scalar_params(2));
        V_phi = V_null .* exp(-lambda.*phi);
        V_prime = - lambda*V_phi;
        rho_phi = (phi_dot^2/(2*CL^2)+V_phi) / (8*pi*GN/CL^2);
        evol_output = [rho_phi V_prime];
        if whether_ini == 1
            x_1_scaling = 2*sqrt(6)/(3*lambda);
            x_2_scaling = 2/(sqrt(3)*lambda);
            
            Omega_phi_ini = 4/lambda^2;
            rho_total_ini = rho_matter_ini / (1-Omega_phi_ini);
            Hubble_rate_ini = sqrt(8*pi*GN/3*rho_total_ini);
            
            phi_ini = 0;
            phi_dot_ini = sqrt(6)*Hubble_rate_ini*x_1_scaling;
            V_null_update = (sqrt(3)*Hubble_rate_ini*x_2_scaling/CL)^2;
            
            ini_output = [phi_ini phi_dot_ini V_null_update];
        else
            ini_output = NaN;
        end
    end
    
    % stepwise scalar field
    if which_model == 3
        [lambda_1, lambda_2, alpha, V_null, N_ini] = deal(scalar_params(1), scalar_params(2), scalar_params(3), scalar_params(4), scalar_params(5));
        lambda = (lambda_1+lambda_2)/2 + (lambda_1-lambda_2)/2.*cos(phi/alpha);
        V_phi = V_null*exp( -(lambda_1+lambda_2)/2.*phi - alpha*(lambda_1-lambda_2)/2.*sin(phi/alpha) );
        V_prime = - lambda*V_phi;
        rho_phi = (phi_dot^2/(2*CL^2)+V_phi) / (8*pi*GN/CL^2);
        evol_output = [rho_phi V_prime];
        if whether_ini == 1
            x_1_start = 0.75;
            x_2_start = 0.5;
            lambda_start = lambda_2+2;
            IC = 1;
            nu_start = sqrt(6)/alpha*sqrt(lambda_start*(lambda_1+lambda_2)-lambda_start^2-lambda_1*lambda_2);
            if IC == 2
                nu_start = - nu_start;
            end
            X_start = [x_1_start x_2_start lambda_start nu_start];
            
            N_span = [0 100];
            options = odeset('RelTol', 1E-10, 'AbsTol', 1E-10, 'MaxStep', 1E-5);
            [N, X_OSS] = ode45(@(N, X) sineoEoS(N, X, scalar_params), N_span, X_start, options);
            
            x_1_ini    = interp1(N, X_OSS(:,1), N_ini, 'spline');
            x_2_ini    = interp1(N, X_OSS(:,2), N_ini, 'spline');
            lambda_ini = interp1(N, X_OSS(:,3), N_ini, 'spline');
            nu_ini     = interp1(N, X_OSS(:,4), N_ini, 'spline');
            
            Omega_phi_ini = x_1_ini^2 + x_2_ini^2;
            rho_total_ini = rho_matter_ini / (1-Omega_phi_ini);
            Hubble_rate_ini = sqrt(8*pi*GN/3*rho_total_ini);
            
            phi_dot_ini = sqrt(6)*Hubble_rate_ini*x_1_ini;
            
            phi_ini = alpha * acos( 2/(lambda_1-lambda_2) * (lambda_ini-(lambda_1+lambda_2)/2) );
            if nu_ini > 0 
                phi_ini = 2*alpha*pi - phi_ini;
            end
            
            V_phi_ini = (sqrt(3)*Hubble_rate_ini*x_2_ini/CL)^2;
            V_null_update = V_phi_ini / exp( - (lambda_1+lambda_2)/2.*phi_ini - alpha*(lambda_1-lambda_2)/2.*sin(phi_ini/alpha) );
            
            ini_output = [phi_ini phi_dot_ini V_null_update];
        else
            ini_output = NaN;
        end
    end
end

function dXdN = sineoEoS(N, X, scalar_params)
    % This is only used when which_model == 3
    lambda_1 = scalar_params(1);
    lambda_2 = scalar_params(2);
    alpha    = scalar_params(3);
    w_rad    = 1/3;
    
    dXdN = zeros(4,1);
    x_1 = X(1);
    x_2 = X(2);
    lambda = X(3);
    nu = X(4);
    
    L = (1-w_rad).*x_1.^2 + (1+w_rad).*(1-x_2.^2);
    
    dXdN(1) = - 3*x_1 + sqrt(6)/2*lambda.*x_2.^2 + 3/2*x_1.*L;
    dXdN(2) = -sqrt(6)/2*lambda.*x_1.*x_2 + 3/2*x_2.*L;
    dXdN(3) = nu.*x_1;
    dXdN(4) = 3*x_1./alpha^2.*(lambda_1+lambda_2-2*lambda);
end
