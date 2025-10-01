/**
 * Service Registry - Manages service registration and discovery
 * Provides a centralized registry for all services in the MCP system
 */

export class ServiceRegistry {
  constructor() {
    this.services = new Map();
    this.metadata = new Map();
    this.dependencies = new Map();
    this.listeners = new Map();
    this.isInitialized = false;
  }

  /**
   * Initialize the service registry
   */
  async initialize() {
    try {
      this.isInitialized = true;
      this.emit('registry:initialized');
      return true;
    } catch (error) {
      throw new Error(`Failed to initialize service registry: ${error.message}`);
    }
  }

  /**
   * Register a service
   */
  async register(serviceName, serviceInstance, metadata = {}) {
    if (!this.isInitialized) {
      throw new Error('Service registry not initialized');
    }

    if (this.services.has(serviceName)) {
      throw new Error(`Service '${serviceName}' is already registered`);
    }

    // Validate service instance
    this.validateService(serviceName, serviceInstance);

    // Store service and metadata
    this.services.set(serviceName, serviceInstance);
    this.metadata.set(serviceName, {
      registeredAt: new Date().toISOString(),
      type: metadata.type || 'unknown',
      version: metadata.version || '1.0.0',
      description: metadata.description || '',
      dependencies: metadata.dependencies || [],
      capabilities: metadata.capabilities || [],
      ...metadata
    });

    // Store dependencies
    if (metadata.dependencies && metadata.dependencies.length > 0) {
      this.dependencies.set(serviceName, metadata.dependencies);
    }

    // Initialize service if it has an initialize method
    if (typeof serviceInstance.initialize === 'function') {
      await serviceInstance.initialize();
    }

    this.emit('service:registered', { serviceName, metadata: this.metadata.get(serviceName) });

    return {
      success: true,
      serviceName,
      metadata: this.metadata.get(serviceName)
    };
  }

  /**
   * Unregister a service
   */
  async unregister(serviceName) {
    if (!this.services.has(serviceName)) {
      throw new Error(`Service '${serviceName}' is not registered`);
    }

    const service = this.services.get(serviceName);

    // Cleanup service if it has a cleanup method
    if (typeof service.cleanup === 'function') {
      await service.cleanup();
    }

    // Remove from registry
    this.services.delete(serviceName);
    this.metadata.delete(serviceName);
    this.dependencies.delete(serviceName);

    this.emit('service:unregistered', { serviceName });

    return { success: true, serviceName };
  }

  /**
   * Get a registered service
   */
  get(serviceName) {
    if (!this.services.has(serviceName)) {
      return null;
    }

    return this.services.get(serviceName);
  }

  /**
   * Get service metadata
   */
  getMetadata(serviceName) {
    return this.metadata.get(serviceName) || null;
  }

  /**
   * Check if a service is registered
   */
  has(serviceName) {
    return this.services.has(serviceName);
  }

  /**
   * List all registered services
   */
  list() {
    const serviceList = [];

    for (const [name, service] of this.services) {
      const metadata = this.metadata.get(name);
      serviceList.push({
        name,
        type: metadata.type,
        version: metadata.version,
        description: metadata.description,
        registeredAt: metadata.registeredAt,
        capabilities: metadata.capabilities,
        isHealthy: this.isServiceHealthy(name)
      });
    }

    return serviceList;
  }

  /**
   * Find services by type
   */
  findByType(type) {
    const services = [];

    for (const [name, service] of this.services) {
      const metadata = this.metadata.get(name);
      if (metadata.type === type) {
        services.push({
          name,
          service,
          metadata
        });
      }
    }

    return services;
  }

  /**
   * Find services by capability
   */
  findByCapability(capability) {
    const services = [];

    for (const [name, service] of this.services) {
      const metadata = this.metadata.get(name);
      if (metadata.capabilities && metadata.capabilities.includes(capability)) {
        services.push({
          name,
          service,
          metadata
        });
      }
    }

    return services;
  }

  /**
   * Get services with dependencies resolved
   */
  getWithDependencies(serviceName) {
    const service = this.get(serviceName);
    if (!service) {
      return null;
    }

    const dependencies = this.dependencies.get(serviceName) || [];
    const resolvedDependencies = {};

    for (const depName of dependencies) {
      const depService = this.get(depName);
      if (!depService) {
        throw new Error(`Dependency '${depName}' for service '${serviceName}' is not registered`);
      }
      resolvedDependencies[depName] = depService;
    }

    return {
      service,
      dependencies: resolvedDependencies,
      metadata: this.metadata.get(serviceName)
    };
  }

  /**
   * Validate service instance
   */
  validateService(serviceName, serviceInstance) {
    if (!serviceInstance) {
      throw new Error(`Service instance for '${serviceName}' cannot be null or undefined`);
    }

    if (typeof serviceInstance !== 'object') {
      throw new Error(`Service instance for '${serviceName}' must be an object`);
    }

    // Optional: Check for required methods based on service type
    // This can be extended based on specific service requirements
  }

  /**
   * Check if a service is healthy
   */
  isServiceHealthy(serviceName) {
    const service = this.get(serviceName);
    if (!service) {
      return false;
    }

    // If service has a health check method, use it
    if (typeof service.isHealthy === 'function') {
      try {
        return service.isHealthy();
      } catch (error) {
        return false;
      }
    }

    // If service has a status method, check status
    if (typeof service.getStatus === 'function') {
      try {
        const status = service.getStatus();
        return status && status.healthy !== false;
      } catch (error) {
        return false;
      }
    }

    // Default: service is healthy if it exists
    return true;
  }

  /**
   * Get overall registry health status
   */
  isHealthy() {
    if (!this.isInitialized) {
      return false;
    }

    // Check if all services are healthy
    for (const serviceName of this.services.keys()) {
      if (!this.isServiceHealthy(serviceName)) {
        return false;
      }
    }

    return true;
  }

  /**
   * Get registry status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      serviceCount: this.services.size,
      services: this.list(),
      healthy: this.isHealthy(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Cleanup registry
   */
  async cleanup() {
    const cleanupPromises = [];

    for (const [serviceName, service] of this.services) {
      if (typeof service.cleanup === 'function') {
        cleanupPromises.push(
          service.cleanup().catch(error =>
            console.error(`Failed to cleanup service ${serviceName}:`, error)
          )
        );
      }
    }

    await Promise.all(cleanupPromises);

    this.services.clear();
    this.metadata.clear();
    this.dependencies.clear();
    this.listeners.clear();

    this.isInitialized = false;
    this.emit('registry:cleaned');
  }

  /**
   * Event system for registry events
   */
  on(event, listener) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(listener);
  }

  /**
   * Remove event listener
   */
  off(event, listener) {
    if (!this.listeners.has(event)) {
      return;
    }

    const listeners = this.listeners.get(event);
    const index = listeners.indexOf(listener);
    if (index > -1) {
      listeners.splice(index, 1);
    }
  }

  /**
   * Emit event
   */
  emit(event, data) {
    if (!this.listeners.has(event)) {
      return;
    }

    const listeners = this.listeners.get(event);
    listeners.forEach(listener => {
      try {
        listener(data);
      } catch (error) {
        console.error(`Error in event listener for ${event}:`, error);
      }
    });
  }

  /**
   * Get dependency graph
   */
  getDependencyGraph() {
    const graph = {};

    for (const [serviceName, deps] of this.dependencies) {
      graph[serviceName] = deps;
    }

    return graph;
  }

  /**
   * Validate dependency graph for circular dependencies
   */
  validateDependencies() {
    const visited = new Set();
    const recursionStack = new Set();

    const hasCycle = (serviceName) => {
      if (recursionStack.has(serviceName)) {
        return true; // Circular dependency found
      }

      if (visited.has(serviceName)) {
        return false;
      }

      visited.add(serviceName);
      recursionStack.add(serviceName);

      const deps = this.dependencies.get(serviceName) || [];
      for (const dep of deps) {
        if (hasCycle(dep)) {
          return true;
        }
      }

      recursionStack.delete(serviceName);
      return false;
    };

    for (const serviceName of this.services.keys()) {
      if (hasCycle(serviceName)) {
        throw new Error(`Circular dependency detected involving service: ${serviceName}`);
      }
    }

    return true;
  }
}

export default ServiceRegistry;