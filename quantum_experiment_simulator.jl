# Импорт необходимых библиотек и пакетов
using LinearAlgebra          # Для работы с линейной алгеброй (матрицы, вектора и т.д.)
using Distributions          # Для генерации случайных чисел из различных распределений
using JSON                   # Для сохранения результатов в формате JSON
using Dates                  # Для работы с датами и временем
using DataFrames             # Для удобной работы с таблицами данных
using Logging                # Для логирования хода выполнения программы
using SparseArrays           # Для работы с разреженными матрицами (не используется в коде, но подключена)
using Flux                   # Библиотека для построения нейросетей
using BlackBoxOptim          # Библиотека для черного оптимизационного поиска (оптимизация без знания градиентов)

# Объявление фундаментальных физических констант
const hbar = 1.0545718e-34   # Постоянная Планка, деленная на 2π (Дж·с)
const G = 6.67430e-11        # Гравитационная постоянная (м³·кг⁻¹·с⁻²)
const c = 299792458.0        # Скорость света в вакууме (м/с)
const kB = 1.380649e-23      # Постоянная Больцмана (Дж/К)

# Определение структуры квантовой системы
struct QuantumSystem
    wavefunction::Vector{ComplexF64}       # Волновая функция вектора состояний (комплексные числа)
    hamiltonian::Matrix{Float64}           # Гамильтониан системы (матрица энергии)
    tunneling::Float64                     # Параметр туннелирования (перехода между состояниями)
    normalization_factor::Float64          # Коэффициент нормализации волновой функции
end

# Функция создания квантовой системы
function create_quantum_system(energies::Vector{Float64}, 
                              coefficients::Vector{ComplexF64}, 
                              tunneling::Float64)
    @assert length(energies) == length(coefficients)  # Проверка: количество энергий = числу коэффициентов
    
    n = length(energies)
    H = diagm(energies)              # Создание диагональной матрицы из энергий (гамильтониан)
    
    # Добавление туннелирования между соседними состояниями (верхняя и нижняя диагональ)
    for i in 1:n-1
        H[i,i+1] = H[i+1,i] = tunneling
    end
    
    norm_factor = max(norm(coefficients), eps(Float64))  # Нормализация коэффициентов (чтобы длина ≠ 0)
    QuantumSystem(coefficients ./ norm_factor, H, tunneling, norm_factor)
end

# Функция эволюции волновой функции во времени
function evolve_state(system::QuantumSystem, t::Float64)
    H = system.hamiltonian           # Получение гамильтониана
    ψ0 = system.wavefunction         # Начальная волновая функция
    t_scaled = t * 1e18              # Перевод времени в аттосекунды
    
    try
        # Расчет унитарного оператора эволюции: U = exp(-iHt/ħ)
        U = exp(-im * H * t_scaled / hbar)
        return U * ψ0                # Применение оператора к волновой функции
    catch e
        @error "Matrix exponential error: $e"   # В случае ошибки возвращаем нулевой вектор
        return zeros(ComplexF64, length(ψ0))
    end
end

# Расчет вероятностей туннелирования по состояниям
function tunneling_probabilities(system::QuantumSystem, t::Float64)
    ψt = evolve_state(system, t)         # Эволюция волновой функции
    probs = abs2.(ψt)                    # Квадрат модуля ψt даёт вероятность каждого состояния
    total = sum(probs)                  # Сумма всех вероятностей
    
    # Нормализация: если сумма конечна и > 0, делим все значения на неё. Иначе — равномерное распределение
    isfinite(total) && total > 0 ? probs ./ total : fill(1/length(probs), length(probs))
end

# Монте-Карло симуляция вероятностей для случайных времен
function monte_carlo_simulation(num_samples::Int, system::QuantumSystem)
    results = Vector{Vector{Float64}}(undef, num_samples)  # Подготовка массива под результаты
    
    for i in 1:num_samples
        t = rand() * 1e-18                     # Случайное время в аттосекундах
        results[i] = tunneling_probabilities(system, t)  # Вероятности для этого времени
    end
    results
end

# Гравитационное воздействие на волновую функцию
function gravitational_effect(ψ::Vector{ComplexF64}, mass::Float64, distance::Float64)
    potential = -G * mass / max(distance, 1e-30)  # Гравитационный потенциал, защита от деления на 0
    ψ .* exp.(im * potential / hbar)              # Модификация фазы волновой функции
end

# Генерация случайных входных данных и целевых значений для обучения
function generate_meta_data(num_tasks::Int)
    inputs = []
    targets = []
    for _ in 1:num_tasks
        energies = rand(3) .* 1e-32                          # Случайные энергии
        coefficients = randn(ComplexF64, 3) .|> x->x/norm(x) # Случайные нормализованные коэффициенты
        push!(inputs, [real.(coefficients); imag.(coefficients); energies])  # Объединяем в вектор
        push!(targets, [mean(energies) * 0.1])               # Цель: усредненная энергия * 0.1
    end
    hcat(inputs...), hcat(targets...)  # Преобразование в матрицы (входы и цели)
end

# Функция потерь: отклонение модели от реальных данных + регуляризация по весам
function loss(re, weights, inputs, targets)
    model = re(weights)  # Восстановление модели из весов
    sum(abs2, model(inputs) .- targets) + 0.1 * norm(weights)  # Ошибка + регуляризация
end

# Главная функция, запускающая весь процесс
function main()
    timestamp = now()  # Текущее время для логов и имени файла
    output_file = "quantum_results_$(Dates.format(timestamp, "yyyy_mm_dd_HH_MM_SS")).json"

    # Включение логирования
    Logging.with_logger(Logging.ConsoleLogger(stdout, Logging.Info)) do
        @info "Начало симуляции"

        # Генерация данных
        X, y = generate_meta_data(1000)
        input_size = size(X, 1)

        # Определение нейросети
        model = Chain(
            Dense(input_size => 64, relu),   # Полносвязный слой 64 нейрона
            Flux.normalize,                  # Нормализация
            Dense(64 => 32, relu),           # Следующий слой
            Flux.normalize,
            Dense(32 => 1)                   # Выходной слой — предсказывает одно число
        ) |> f64

        # Получение вектора весов и функции восстановления модели из них
        initial_weights, re = Flux.destructure(model)
        num_dimensions = length(initial_weights)

        # Оптимизация весов модели с помощью BlackBoxOptim
        result = bboptimize(weights -> loss(re, weights, X, y); 
            SearchRange = (-1.0, 1.0),       # Диапазон значений весов
            NumDimensions = num_dimensions, # Число параметров
            MaxSteps = 1000,                 # Максимальное число шагов
            TraceInterval = 100.0,           # Интервал логов
            PopulationSize = 50)             # Размер популяции в оптимизации

        best_weights = best_candidate(result)  # Лучшие найденные веса
        model = re(best_weights)              # Восстановление модели с лучшими весами

        # Пример квантовой системы
        energies = Float64[1e-32, 2e-32, 3e-32]
        coefficients = ComplexF64[1.0+0im, 0.0+1im, 1.0+1im]
        input_vec = [real.(coefficients); imag.(coefficients); energies]

        # Предсказание туннелирования
        tunneling = clamp(model(input_vec)[1], 1e-35, 1e-30)

        # Создание системы
        system = create_quantum_system(energies, coefficients, tunneling)
        @info "Начальная волновая функция: $(system.wavefunction)"

        # Эволюция во времени
        t = 1e-18
        ψt = evolve_state(system, t)
        @info "Эволюция волновой функции: $ψt"

        # Эффект декогеренции
        γ = 1e-12
        ψ_decayed = decoherence(ψt, system.hamiltonian, t, γ)
        @info "Коофициент декогенеренции: $ψ_decayed"

        # Влияние гравитации
        ψ_grav = gravitational_effect(ψ_decayed, 5.972e24, 1e-3)
        @info "Параметры после воздействия гравитации: $ψ_grav"

        # Монте-Карло моделирование
        mc_results = monte_carlo_simulation(1000, system)
        mc_df = DataFrame(hcat(mc_results...)', :auto)  # Преобразуем в таблицу
        rename!(mc_df, ["State_$i" for i in 1:3])        # Названия столбцов

        # Проверка на невалидные значения
        if any(!isfinite, Matrix(mc_df))
            @error "Invalid values detected"
        else
            @info "Монте-карло результаты:\n$(first(mc_df, 5))"
        end

        # Формирование словаря с результатами и сохранение в файл
        data = Dict(
            "timestamp" => Dates.format(timestamp, "yyyy-mm-dd HH:MM:SS"),
            "initial_wavefunction" => system.wavefunction,
            "final_wavefunction" => ψ_grav,
            "tunneling_strength" => tunneling,
            "monte_carlo_data" => Matrix(mc_df),
            "hamiltonian_matrix" => system.hamiltonian
        )
        open(output_file, "w") do f
            JSON.print(f, data, 4)
        end
        @info "Данные сохранены в $output_file"
    end
end

# Функция декогеренции — моделирует затухание квантовой суперпозиции со временем
function decoherence(ψ::Vector{ComplexF64}, H::Matrix{Float64}, t::Float64, γ::Float64)
    decay = exp(-γ * t)                                     # Экспоненциальное затухание
    QuantumSystem(ψ .* decay, H, 0.0, 1.0) |> s->evolve_state(s, t)  # Эволюция уже затухающего состояния
end

# Вызов основной функции
main()