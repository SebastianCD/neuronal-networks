%Declaración de los puntos presentes en la imagen%
puntos = [4 4;2 7;3 9;6 11;7 4;5 7;8 6;8 9;10 6;10 9];

%Declaración de los grupos para cada punto%
t = [0;0;0;0;0;1;1;1;1;1];

[numRows,numCols] = size(puntos);

%Definimos el número de epocas%
epochs = 2000;

%Se declaran los pesos y bias%
w1 = [-0.9 0.1;0.35 -0.55];
b1 = [-1 -1];
w2 = [-0.7 0.2];
b2 = [-1];

%Iniciamos el algoritmo de aprendizaje%
for epoch = 1: epochs
    err = 0;
    count = 1;
    for i = 1: numRows
        punto = [puntos(i,1) puntos(i,2)];
        %Hacia adelante%
        [a1, a2] = ff(punto, w1, b1, w2, b2);
        %Calculo de error en la red%
        err = nrror(t(count), a2);
        %Propagación hacia atras%
        [w1, b1, w2, b2] = bp(punto, t(count), w1, b1, w2, b2, a1, a2);
        count = count + 1;
    end
end

Z = ones(numRows,1);

%Convertir problema lineal a no lineal%
for x = 1: numRows
    punto = [puntos(x,1) puntos(x,2)];
    [X, Y] = ff(punto,w1,b1,w2,b2);
    Z(x) = Y;
end

fprintf('Patrón:    t:    Salida:\n');
for i = 1: numRows
    fprintf('[%d,%d]       %d       %.3f\n',puntos(i,1),puntos(i,2),t(i),Z(i));
end

fprintf('\nPesos de la capa intermedia:\n');
disp(w1)

fprintf('Biases de las capas intermedias:\n');
disp(b1)

fprintf('Pesos de la capa de salida:\n');
disp(w2)

fprintf('Bias de la capa de salida:\n');
disp(b2)

%Propagación hacia atras%
function [nw1, nb1, nw2, nb2] = bp(punto, t, w1, b1, w2, b2, a1, a2)
    %Se define el factor de aprendizaje%
    alpha = 0.25;
    L_error = -(t - a2) * a2 * (1 - a2);
    nw2 = w2 - alpha * L_error * a1;
    nb2 = b2 - alpha * L_error;
    l_error = L_error .* w2 .* a1 .* (1 - a1);
    nb1 = b1 - alpha * l_error;
    aux = [punto(1,1) ; punto(1,2)];
    l_error2 = [l_error(1,1); l_error(1,2)];
    nw1 = w1 - alpha * aux .* l_error2;
end

%Sigmoidal%
function [s] = sig(z)
    aux = exp(-z);
    s = 1 ./ (1 + aux);
end

%Propagación hacia adelante%
function [a,a2] = ff(punto, w1, b1, w2, b2)
    aux = transpose(w1);
    z2 = transpose(aux*punto.') + b1;
    a = sig(z2);
    z3 = dot(a,w2) + b2;
    a2 = sig(z3);
end

%Calculo de error en la red%
function [err] = nrror(t,a)
    err = 0.5 * (t - a).^2;
end