import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

# creation dossier resultats qui contiendra les differents graphes
output_dir = "resultats"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#creation de matrice de distances aléatoires
def create_distance_matrix(n):
    matrix = np.random.randint(10, 100, size=(n, n))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    return matrix

# Calcul de distance totale d'un circuit
def calculate_distance(tour, distance_matrix):
    distance = 0
    for i in range(len(tour)):
        distance += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return distance

#  -----------------------------------------------------------------------------------------
#   Algorithme génétique (AG)
#  -----------------------------------------------------------------------------------------

# 1- Creation de population initiale
def create_population(pop_size, n_cities):
    population = []
    for _ in range(pop_size):
        tour = list(range(n_cities))
        random.shuffle(tour)
        population.append(tour)
    return population

# 2- selection par tournoi
def tournament_selection(population, distance_matrix, tournament_size):
    tournament = random.sample(population, tournament_size)
    distances = [calculate_distance(tour, distance_matrix) for tour in tournament]
    return tournament[np.argmin(distances)]

# 3- Croisement OX
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    remaining = [city for city in parent2 if city not in child[start:end+1]]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    return child

# 4- mutation par echange
def swap_mutation(tour, mutation_rate):
    tour = tour.copy()
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

# optimisation 2-opt 
def two_opt(tour, distance_matrix):
    best_tour = tour.copy()
    best_distance = calculate_distance(best_tour, distance_matrix)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 2, len(tour)):
                new_tour = best_tour.copy()
                new_tour[i:j] = best_tour[j-1:i-1:-1]
                new_distance = calculate_distance(new_tour, distance_matrix)
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
    
    return best_tour, best_distance

# Algorithme génétique
def genetic_algorithm(distance_matrix, pop_size=50, generations=100, mutation_rate=0.1, tournament_size=5):
    n_cities = len(distance_matrix)
    population = create_population(pop_size, n_cities)
    best_tour = None
    best_distance = float('inf')
    
    for generation in range(generations):
        new_population = []
        # trouver le meilleur individu et applique optimisation 2-opt uniquement au meilleur indiv
        best_in_gen = trouveMeilleurIndividu(distance_matrix, pop_size, mutation_rate, tournament_size, population, new_population)
        optimized_tour, optimized_distance = two_opt(best_in_gen, distance_matrix)
        if optimized_distance < best_distance:
            best_distance = optimized_distance
            best_tour = optimized_tour.copy()
        population = new_population
        
    return best_tour, best_distance

def trouveMeilleurIndividu(distance_matrix, pop_size, mutation_rate, tournament_size, population, new_population):
    distances = [calculate_distance(tour, distance_matrix) for tour in population]
    best_idx = np.argmin(distances)
    best_in_gen = population[best_idx]
    best_in_gen_distance = distances[best_idx]
        
    for _ in range(pop_size):
        parent1 = tournament_selection(population, distance_matrix, tournament_size)
        parent2 = tournament_selection(population, distance_matrix, tournament_size)
        child = order_crossover(parent1, parent2)
        child = swap_mutation(child, mutation_rate)
        new_population.append(child)
    return best_in_gen

# -----------------------------------------------------------------------------------------
# --- Algorithme des colonies de fourmis (ACO) ---

def ant_colony_optimization(distance_matrix, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
    n_cities = len(distance_matrix)
    pheromone = np.ones((n_cities, n_cities)) / n_cities
    best_tour = None
    best_distance = float('inf')
    no_improvement_count = 0
    # Arrêter si apres 20 iterations pas d'amelioration
    max_no_improvement = 20  
    
    for iteration in range(n_iterations):
        all_tours = []
        all_distances = []
        
        for ant in range(n_ants):
            tour = [(ant % n_cities)]
            unvisited = set(range(n_cities)) - set(tour)
            
            while unvisited:
                current_city = tour[-1]
                proba = []
                for next_city in unvisited:
                    tau = pheromone[current_city][next_city] ** alpha
                    eta = (1.0 / distance_matrix[current_city][next_city]) ** beta
                    proba.append(tau * eta)
                proba = np.array(proba)
                proba /= proba.sum()
                
                next_city = random.choices(list(unvisited), weights=proba, k=1)[0]
                tour.append(next_city)
                unvisited.remove(next_city)
            
            distance = calculate_distance(tour, distance_matrix)
            all_tours.append(tour)
            all_distances.append(distance)
            
            if distance < best_distance:
                best_distance = distance
                best_tour = tour.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        
        pheromone *= (1 - evaporation_rate)
        for tour, distance in zip(all_tours, all_distances):
            for i in range(len(tour)):
                city_i = tour[i]
                city_j = tour[(i + 1) % n_cities]
                pheromone[city_i][city_j] += Q / distance
                pheromone[city_j][city_i] += Q / distance
        
        # arret si aucune amélioration
        if no_improvement_count >= max_no_improvement:
            break
    
    # applique 2-opt au meilleur circuit
    best_tour, best_distance = two_opt(best_tour, distance_matrix)
    
    return best_tour, best_distance

# Visualisation du circuit
def plot_tour(tour, distance_matrix, n_cities, filename):
    try:
        coords = np.random.rand(n_cities, 2) * 100
        plt.scatter(coords[:, 0], coords[:, 1], c='blue')
        for i, coord in enumerate(coords):
            plt.text(coord[0], coord[1], f"{i}")
        for i in range(len(tour)):
            start = coords[tour[i]]
            end = coords[tour[(i + 1) % len(tour)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')
        plt.title(f"Circuit avec distance {calculate_distance(tour, distance_matrix):.2f}")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Image enregesitree dans '{os.path.join(output_dir, filename)}'")
    except Exception as e:
        print(f"erreur lors de la creation de l'image {filename} : {e}")

# Graphe de comparaison des distances
def plot_distance_comparison(ag_results, aco_results):
    n_cities_list = [5, 10, 15, 50]
    ag_distances = [result['best_distance'] for result in ag_results]
    aco_distances = [result['best_distance'] for result in aco_results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(n_cities_list, ag_distances, marker='o', label='AG', color='blue')
    plt.plot(n_cities_list, aco_distances, marker='o', label='ACO', color='red')
    plt.xlabel('NBre de villes (n_cities)')
    plt.ylabel('Distance')
    plt.title('Comparaison des distances entre AG et ACO')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "distance_comparison.png"))
    plt.close()
    print(f"graphe de comparaison des distances enregesitree dans '{os.path.join(output_dir, 'distance_comparison.png')}'")

# Graphe de comparaison des temps d'exécution
def plot_time_comparison(ag_results, aco_results):
    n_cities_list = [5, 10, 15, 50]
    ag_times = [result['execution_time'] for result in ag_results]
    aco_times = [result['execution_time'] for result in aco_results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(n_cities_list, ag_times, marker='o', label='AG', color='blue')
    plt.plot(n_cities_list, aco_times, marker='o', label='ACO', color='red')
    plt.xlabel('Nombre de villes (n_cities)')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.title('Comparaison des temps d\'exécution entre AG et ACO')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()
    print(f"Graphe de comparaison des temps sauvegardé dans '{os.path.join(output_dir, 'time_comparison.png')}'")


# ----------------------------------------------------------------------------------------------------------
def main():
    n_cities_list = [5, 10, 15, 50]
    ag_results = []
    aco_results = []
    matrices = {}
    
    # genere matrices des distances
    for n_cities in n_cities_list:
        matrices[n_cities] = create_distance_matrix(n_cities)
    
    # ouvrir un fichier pour sauvegarder les resultats
    with open(os.path.join(output_dir, "resultats.txt"), "w") as f:
        f.write("Résultats combinés pour algos AG et ACO \n\n")
        
        for n_cities in n_cities_list:
            distance_matrix = matrices[n_cities]
            num_executions = 1 if n_cities == 50 else 3  # Une seule exécution pour n_cities=50 (optimisation temps)
            
            # --- Resultats Algorithme génétique ---
            print(f"\n test avec n_cities = {n_cities} (AG) ")
            f.write(f"test avec n_cities = {n_cities} (AG) \n")
            
            start_time = time.time()
            best_distance_overall = float('inf')
            best_tour_overall = None
            for i in range(num_executions):
                print(f"\nExécution {i+1}")
                f.write(f"Exécution {i+1}\n")
                best_tour, best_distance = genetic_algorithm(distance_matrix)
                print(f"Meilleur circuit est : {best_tour}, Distance : {best_distance}")
                f.write(f"Meilleur circuit est : {best_tour}, Distance : {best_distance}\n")
                
                if best_distance < best_distance_overall:
                    best_distance_overall = best_distance
                    best_tour_overall = best_tour
            
            execution_time = time.time() - start_time
            print("\nMatrice des distances :")
            print(distance_matrix)
            print(f"meilleure solution pour n_cities={n_cities} : {best_tour_overall}, Distance : {best_distance_overall}")
            print(f"temps d'exécution total : {execution_time:.2f} secondes")
            
            f.write("\nMatrice des distances :\n")
            f.write(np.array2string(distance_matrix) + "\n")
            f.write(f"Meilleure solution pour n_cities={n_cities} : {best_tour_overall}, Distance : {best_distance_overall}\n")
            f.write(f"Temps d'exécution total : {execution_time:.2f} secondes\n\n")
            
            filename = f"tour_{n_cities}.png"
            plot_tour(best_tour_overall, distance_matrix, n_cities, filename)
            
            ag_results.append({
                "n_cities": n_cities,
                "best_tour": best_tour_overall,
                "best_distance": best_distance_overall,
                "execution_time": execution_time
            })
            
            #Algorithme des colonies de fourmis (ACO) 
            print(f"\n=== Test avec n_cities = {n_cities} (ACO) ===")
            f.write(f"=== Test avec n_cities = {n_cities} (ACO) ===\n")
            
            start_time = time.time()
            best_distance_overall = float('inf')
            best_tour_overall = None
            for i in range(num_executions):
                print(f"\nExécution {i+1}")
                f.write(f"Exécution {i+1}\n")
                best_tour, best_distance = ant_colony_optimization(distance_matrix, n_ants=20, n_iterations=100)
                print(f"Meilleur circuit : {best_tour}, Distance : {best_distance}")
                f.write(f"Meilleur circuit : {best_tour}, Distance : {best_distance}\n")
                
                if best_distance < best_distance_overall:
                    best_distance_overall = best_distance
                    best_tour_overall = best_tour
            
            execution_time = time.time() - start_time
            print("\nMatrice des distances :")
            print(distance_matrix)
            print(f"Meilleure solution pour n_cities={n_cities} : {best_tour_overall}, Distance : {best_distance_overall}")
            print(f"Temps d'exécution total : {execution_time:.2f} secondes")
            
            f.write("\nMatrice des distances :\n")
            f.write(np.array2string(distance_matrix) + "\n")
            f.write(f"Meilleure solution pour n_cities={n_cities} : {best_tour_overall}, Distance : {best_distance_overall}\n")
            f.write(f"Temps d'exécution total : {execution_time:.2f} secondes\n\n")
            
            filename = f"tour_aco_{n_cities}.png"
            plot_tour(best_tour_overall, distance_matrix, n_cities, filename)
            
            aco_results.append({
                "n_cities": n_cities,
                "best_tour": best_tour_overall,
                "best_distance": best_distance_overall,
                "execution_time": execution_time
            })
        
        print("\n------------------------------------------------")
        print("\nRésumé des résultats ")
        f.write("=== Résumé des résultats ===\n")
        f.write("Algorithme génétique (AG):\n")
        for result in ag_results:
            summary = f"n_cities={result['n_cities']}: Distance={result['best_distance']:.2f}, Temps={result['execution_time']:.2f}s, Circuit={result['best_tour']}"
            print(f"AG - {summary}")
            f.write(f"{summary}\n")
        
        f.write("\nAlgorithme des colonies de fourmis (ACO):\n")
        for result in aco_results:
            summary = f"n_cities={result['n_cities']}: Distance={result['best_distance']:.2f}, Temps={result['execution_time']:.2f}s, Circuit={result['best_tour']}"
            print(f"ACO - {summary}")
            f.write(f"{summary}\n")
    
    # genere graphes de comparaison
    plot_distance_comparison(ag_results, aco_results)
    plot_time_comparison(ag_results, aco_results)

if __name__ == "__main__":
    main()