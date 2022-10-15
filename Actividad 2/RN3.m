%Declaración de los puntos presentes en la imagen%
puntos = [2 6; 4 4; 6 3; 4 10; 7 10; 9 8];

%Declaración de los grupos para cada punto%
t = [-1; -1; -1; 1; 1; 1];

%Declaración de bias para cada punto%
bias = ones(6,1)*(-1);

%Agregamos el bias a cada punto%
puntos = [puntos bias];

%Se declaran los pesos%
w = [0.5 0.35 0.9];

%Producto punto entre el vector de puntos y los pesos%
a = puntos * w.';

%Se define el factor de aprendizaje%
alpha = 0.02;

[numRows,numCols] = size(puntos);

%Relizamos la operación sigmoidal%
y = 1/(1+exp(-a));

%Definimos el número de epocas%
epoch = 0;
epochs = 300;

figure(1)
grid on;
hold on;
xlim([-1 11])
ylim([-1 11])
%Graficamos la frontera de decisión con los pesos originales%
x = -1:1:11;
front = w(3)/w(2) - x*w(1)/w(2);
plot(x,front);

%Graficamos los puntos%
for i = 1:numRows
    if t(i) == -1
        plot(puntos(i,1),puntos(i,2),'b*');
    else
        plot(puntos(i,1),puntos(i,2),'r*');
    end
end

%Iniciamos el algoritmo de aprendizaje%
for i = 1:epochs
    epoch = epoch + 1;
    for j = 1:numRows
        punto = [puntos(j,1) puntos(j,2) puntos(j,3)];
        %Realizamos el producto del punto y los pesos%
        a = dot(punto,w);
        %Realizamos la operacion sigmoidal%
        y(j)= 1/(1+exp(-a));
        %Calculamos los nuevos pesos%
        wn = w - alpha * y(j)*(1-y(j))*(y(j)-t(j)) * punto;
        w = wn;
    end
end

figure(2)
grid on;
hold on;
xlim([-1 11])
ylim([-1 11])

%Graficamos la frontera de decisión%
x = -1:1:11;
front = w(3)/w(2) - x*w(1)/w(2);
plot(x,front);

%Graficamos los puntos%
for i = 1:numRows
    if t(i) == -1
        plot(puntos(i,1),puntos(i,2),'b*');
    else
        plot(puntos(i,1),puntos(i,2),'r*');
    end
end

%Imprimimos los pesos finales%
fprintf('\nPesos finales:\n');
disp(w)

%Declaración de los puntos de prueba%
puntos = [5 5;6 8];
bias = ones(2,1)*(-1);

%Agregamos el bias a cada punto%
puntos = [puntos bias];
[numRows,numCols] = size(puntos);

for i = 1:numRows
    punto = [puntos(i,1) puntos(i,2) puntos(i,3)];
    %Realizamos el producto del punto y los pesos%
    a = dot(punto,w);
    %Realizamos la operacion hardlim%
    m = hardlim(a);
    fprintf('Para el punto [%d, %d]: %d\n',punto(1),punto(2),m);
end